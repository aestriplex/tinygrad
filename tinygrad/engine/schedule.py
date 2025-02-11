import sys, functools, atexit, pickle
from collections import defaultdict, deque
from dataclasses import dataclass, field
from tinygrad.ops import UOp, Variable, Ops, GroupOp, PatternMatcher, UPat, graph_rewrite, graph_rewrite_map, track_rewrites, buffers
from tinygrad.ops import can_pad, identity_element, resolve, symbolic_simple, view_left, merge_views
from tinygrad.helpers import Context, ContextVar, Metadata, all_int, all_same, colored, diskcache_put, prod, dedup, unwrap, flatten
from tinygrad.helpers import FUSE_CONV_BW, FUSE_ARANGE, DEBUG, CAPTURE_PROCESS_REPLAY, DONT_REALIZE_EXPAND
from tinygrad.dtype import ImageDType
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View, strides_for_shape
from tinygrad.device import Buffer
from tinygrad.spec import type_verify, kernel_spec

# creation can recurse a lot
sys.setrecursionlimit(10000)

# **** schedule simplifier

def simplify_stride0_reduce(reduce:UOp, x:UOp):
  # must be unmasked (NOTE: can be relaxed if not masked on stride 0 axis)
  if any(v.mask is not None for v in unwrap(x.st).views): return None
  # must have all stride 0 in the relevant axis (NOTE: can do partial)
  if not all(unwrap(x.st).views[-1].strides[axis] == 0 for axis in reduce.arg[1]) or not all_int(x.shape): return None
  prshape = prod(x.shape[i] for i in reduce.arg[1])
  ret = x.shrink(tuple((0,s) if i not in reduce.arg[1] else (0,1) for i,s in enumerate(x.shape)))
  match reduce.arg[0]:
    case Ops.ADD: return ret*prshape
    case Ops.MUL: return ret.pow(prshape)
    case Ops.MAX: return ret # NOTE: Ops.MAX is passthrough

def found_contiguous(ctx:dict[UOp, UOp], contig:UOp, src:UOp):
  if (sti:=unwrap(src.st).invert(src.base.shape)) is not None: ctx[src.base] = contig.view(sti)
def replace_contiguous(ctx:dict[UOp, UOp], alu:UOp):
  new_src = list(alu.src)
  for i,s in enumerate(alu.src):
    if (replace_src:=ctx.get(s, None)) is not None: new_src[i] = replace_src
  if tuple(new_src) != alu.src: return alu.replace(src=tuple(new_src))

sym = symbolic_simple+PatternMatcher([
  # UOp with size 0 is zero
  (UPat(GroupOp.All-{Ops.SINK}, name="root"), lambda root: root.const_like(0) if root.base.st is not None and root.size == 0 \
    and not (root.base.op is Ops.CONST and root.base.arg == 0) else None),
  # DETACH and CONTIGUOUS_BACKWARD are NOOPs here
  (UPat((Ops.DETACH, Ops.CONTIGUOUS_BACKWARD), name="x"), lambda x: x.src[0]),
  # reduce of size 0 is the identity element
  (UPat(Ops.REDUCE_AXIS, name="reduce", src=(UPat.var("x"),)),
   lambda reduce,x: reduce.const_like(identity_element(reduce.arg[0], reduce.dtype)) if x.size == 0 and reduce.size != 0 else None),
  # reduce on stride 0 is collapsed
  (UPat(Ops.REDUCE_AXIS, name="reduce", src=(UPat.var("x"),)), simplify_stride0_reduce),
  # COPY(CONST) creates a new CONST on the destination device
  (UPat(Ops.COPY, name="root", src=(UPat(), UPat.cvar("x"),)), lambda root,x: root.const_like(x.arg)),
  # no COPY to same device, except clone (arg is True)
  (UPat(Ops.COPY, src=(UPat(), UPat.var("copyin")), name="copy"),
   lambda copyin,copy: copyin if copyin.device == copy.device and copy.arg is not True else None),
  # remove cast to image when it's already a contiguous image
  (UPat(Ops.VIEW, name="vm1", src=(UPat(Ops.CAST, name="cast", src=(UPat(Ops.VIEW, name="vm2", src=(UPat(Ops.CONTIGUOUS, name="base"))))),)),
   lambda cast,base,vm1,vm2: base.view(vm2.st+vm1.st) if isinstance(cast.dtype, ImageDType) and isinstance(base.dtype, ImageDType) else None),
  # remove contiguous if we can just view the buffer
  (UPat(Ops.CONTIGUOUS, name="root", src=(UPat(Ops.VIEW, name="view", src=(UPat(Ops.BUFFER, name="buf"),)),)),
   lambda root,view,buf: view if view.st.contiguous and view.size == buf.size else None),
  # contiguous/buffer/copy is already contiguous
  (UPat(Ops.CONTIGUOUS, name="root", src=(UPat((Ops.CONTIGUOUS, Ops.BUFFER, Ops.COPY)),)), lambda root: root.src[0]),
  # support for using a contiguous permuted view instead of the parent view if one exists
  (UPat(Ops.CONTIGUOUS, name="contig", src=(UPat(Ops.VIEW, name="src"),)), found_contiguous),
  (UPat(GroupOp.ALU, name="alu"), replace_contiguous),
  # substitute BITCAST/CONTIGUOUS with BUFFER_VIEW on DISK
  (UPat((Ops.BITCAST, Ops.CONTIGUOUS), name="root"),
  lambda root: root.replace(op=Ops.BUFFER_VIEW) if isinstance(root.device, str) and root.device.startswith("DISK") else None),
  # remove CONST/BIND/BUFFER/VIEW from SINK
  (UPat(Ops.SINK, name="root"),
    lambda root: UOp(Ops.SINK, root.dtype, new_src, root.arg)
      if (new_src:=tuple(x.base for x in root.src if not x.is_realized and x.base.op not in {Ops.CONST, Ops.BIND})) != root.src else None),
])

remove_movement_ops = merge_views+PatternMatcher([
  # NOTE: movement ops are always applied to base
  (UPat(GroupOp.Movement, name="mov", src=(UPat.var("x"),)), lambda x,mov: x.view(unwrap(mov.st))),
  # some masked views can collapse to 0, VIEW(x) -> CONST(VIEW)
  (UPat(Ops.VIEW, name="view"),
   lambda view: view.const_like(0) if (vm:=view.st.views[-1].mask) is not None and any((x[1]-x[0]) == 0 for x in vm) else None),
])

# **** ScheduleItem

@dataclass(frozen=True)
class ScheduleItem:
  ast: UOp
  bufs: tuple[Buffer, ...]
  metadata: tuple[Metadata, ...] = ()

# **** schedule creation and toposort

class Kernel:
  def __init__(self, ast:UOp): self.ast = ast
  def __repr__(self): return f"{self.ast.op}"

DONT_PLACE_IN_KERNEL = {Ops.KERNEL, Ops.BUFFER, Ops.DEVICE}

def append_to_kernel(x:UOp):
  new_srcs: list[UOp] = []
  for s in x.src:
    if s.op in DONT_PLACE_IN_KERNEL: new_srcs.append(s)
    else: new_srcs.extend(s.src)
  return x.replace(src=n) if (n:=tuple(dedup(new_srcs))) != x.src else None

create_kernels = PatternMatcher([
  (UPat(Ops.SINK, name="x"),
   lambda x: x.replace(src=n) if (n:=tuple(s if s.op is Ops.KERNEL else UOp(Ops.KERNEL, src=s.src, arg=Kernel(s)) for s in x.src))!=x.src else None),
  (UPat((Ops.COPY, Ops.CONTIGUOUS), name="x"), lambda x: UOp(Ops.KERNEL, src=x.src, arg=Kernel(x))),
  (UPat(Ops.KERNEL, name="x"), append_to_kernel),
])

assign_bufs = PatternMatcher([
  (UPat(Ops.KERNEL, name="k"), lambda k: UOp.new_buffer((x:=k.arg.ast).device, x.size, x.dtype)),
])

def load_buffer(ctx:list[UOp], x:UOp):
  assert x not in ctx
  ctx.append(x)
  return UOp.load(UOp(Ops.DEFINE_GLOBAL, x.dtype.ptr(x.size), (), len(ctx)-1), unwrap(x.st).to_uop(), dtype=x.dtype)

fix_kernel_ops = PatternMatcher([
  (UPat(Ops.BUFFER, name="x"), load_buffer),
  (UPat(Ops.SINK, src=(UPat(Ops.COPY, name="s"),)), lambda s:s),
  (UPat(Ops.SINK, src=(UPat(GroupOp.All-{Ops.STORE}, name="s"),)),
   lambda s:UOp.store(UOp(Ops.DEFINE_GLOBAL, s.dtype.ptr(s.size), (), 0), ShapeTracker.from_shape(s.shape).to_uop(), s).sink()),
  (UPat(Ops.CONTIGUOUS, src=(UPat.var("s"),)), lambda s:s),
])

@track_rewrites(named=True)
def create_schedule_with_vars(big_sink:UOp) -> tuple[list[ScheduleItem], dict[Variable, int], dict[UOp, UOp]]:
  tensor_map = graph_rewrite_map(big_sink, remove_movement_ops+sym, ctx={})
  # tensors can become an existing buffer or simplify to a const, no ScheduleItem needed
  becomes_map: dict[UOp, UOp] = {}
  for k,v in tensor_map.items():
    if k is v: continue # NOOP
    if v.base.op is Ops.BUFFER:
      # backtrack to the realized tensor
      buf_src = [x for x in k.toposort if (xs:=tensor_map[x]).base is v.base and xs.st == v.st]
      if k is not buf_src[0]: becomes_map[k] = buf_src[0]
    if v.op is Ops.CONST and all_int(v.shape): becomes_map[k] = v

  # discover which of the remaining uops need to realize
  sink = tensor_map[big_sink]
  while 1:
    kernel_map = graph_rewrite_map(sink, create_kernels)
    tensor_map.update(kernel_map)
    sink = kernel_map[sink]
    rep: dict[UOp, UOp] = {}
    for x in sink.toposort:
      if x.op is not Ops.KERNEL: continue
      for s in x.src:
        if s.op in DONT_PLACE_IN_KERNEL: continue
        rep[s] = UOp(Ops.KERNEL, src=s.src, arg=Kernel(s))
    if len(rep) == 0: break
    sink = sink.substitute(rep)

  # map kernels to buffers
  buffer_map = graph_rewrite_map(sink, assign_bufs)

  # prepare maps for tensor realization
  buffer_map = {k:v for k,v in buffer_map.items() if k is not v}
  rev_tensor_map: dict[UOp, list[UOp]] = {}
  for k,v in tensor_map.items(): rev_tensor_map.setdefault(v, []).append(k)

  # linearize
  schedule: list[ScheduleItem] = []
  var_vals: dict[Variable, int] = {}
  for x in sink.toposort:
    if x.op is not Ops.KERNEL: continue
    for tensor_uop in rev_tensor_map[x]: becomes_map[tensor_uop] = buffer_map[x].reshape(tensor_uop.shape)
    ast = graph_rewrite(x.arg.ast.substitute(buffer_map).sink(), fix_kernel_ops, global_bufs:=[buffer_map[x]])
    schedule.append(ScheduleItem(ast, tuple(b.buffer for b in global_bufs)))

  return schedule, var_vals, becomes_map
