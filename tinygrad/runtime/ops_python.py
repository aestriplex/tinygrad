# pylint: disable=cell-var-from-loop
# a python uops emulator
# works to test the tensor cores, and all the uops in general
# this is the (living) definition of uops
from typing import Tuple, List, Optional, Any, Dict
import pickle, base64, itertools, time, struct
from tinygrad.codegen.kernel import TensorCoreLayout
from tinygrad.dtype import DType, dtypes, ImageDType
from tinygrad.helpers import all_same, getenv, flatten
from tinygrad.device import Compiled, Compiler, Allocator
from tinygrad.ops import BinaryOps, TernaryOps, exec_alu, truncate, UOps, UOp
from tinygrad.renderer import Renderer
from tinygrad.renderer.cstyle import CUDARenderer, MetalRenderer, AMDRenderer, IntelRenderer, ClangRenderer

def _load(m, i):
  if i < 0 or i >= len(m): raise IndexError(f"load out of bounds, size is {len(m)} and access is {i}")
  return m[i]

def load(inp, j=0):
  if len(inp) == 4: return [_load(m, x+j) if gate else default for m,x,default,gate in zip(*inp)]
  return [_load(m, x+j) for m,x in zip(inp[0], inp[1])]

def _store(m, i, v):
  if i < 0 or i >= len(m): raise IndexError(f"store out of bounds, size is {len(m)}, access is {i}, value is {v}")
  m[i] = v

class PythonProgram:
  def __init__(self, name:str, lib:bytes):
    self.uops: List[Tuple[UOps, Optional[DType], List[int], Any]] = pickle.loads(lib)
  def __call__(self, *bufs, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1), vals:Tuple[int, ...]=(), wait=False):
    st = time.perf_counter()
    warp = list(itertools.product(*[range(x) for x in local_size[::-1]]))
    warp_size = len(warp)
    for idxs in itertools.product(*[range(x) for x in global_size[::-1]]):
      ul: Dict[int, Any] = {}
      dl: Dict[int, DType] = {}
      pbufs: List[memoryview] = list(bufs)
      pvals: List[int] = list(vals)
      i = 0
      loop_ends: Dict[int, int] = {}
      while i < len(self.uops):
        uop, dtype, idp, arg = self.uops[i]
        void_ops = {UOps.STORE, UOps.ENDRANGE, UOps.BARRIER, UOps.IF, UOps.ENDIF}
        if uop is UOps.DEFINE_ACC: idp = [idp[0]]
        inp = [ul[v] for v in idp if self.uops[v][0] not in void_ops]
        dtp = [dl[v] for v in idp if self.uops[v][0] not in void_ops]
        tcl = TensorCoreLayout(warp_size,inp)
        if getenv("TRACE"): print(i, uop, dtype, arg, inp, dtp)
        if uop is UOps.STORE:
          if len(inp) == 3: inp.append([True] * len(inp[0]))  # set the gate to True
          if isinstance(dtp[0], ImageDType):
            # image store
            assert dtp[2].count == 4
            for j,val in enumerate(inp[2]):
              for m,ox,oy,v,g in zip(inp[0], inp[1][0], inp[1][1], val, inp[3]):
                assert ox >= 0 and ox < dtp[0].shape[1] and oy >= 0 and oy < dtp[0].shape[0]
                if g: _store(m, ox*4 + oy*dtp[0].shape[1]*4 + j, v)
          elif dtp[2].count > 1:
            for j,val in enumerate(inp[2]):
              for m,o,v,g in zip(inp[0], inp[1], val, inp[3]):
                if g: _store(m, o+j, v)
          else:
            for m,o,v,g in zip(*inp):
              if g: _store(m, o, v)
          i += 1
          continue
        if uop is UOps.ENDRANGE:
          loop_ends[idp[0]] = i
          i = idp[0]
          continue
        if uop in (UOps.BARRIER, UOps.IF, UOps.ENDIF):
          # in the python emulator, the warp is always in sync
          i += 1
          continue
        assert dtype is not None, f"{uop} is missing a dtype"
        dl[i] = dtype
        if uop is UOps.DEFINE_GLOBAL:
          assert dtype.fmt is not None
          ul[i] = [pbufs.pop(0).cast(dtype.fmt)] * warp_size
        elif uop is UOps.DEFINE_LOCAL:
          assert dtype.fmt is not None
          lbuf = memoryview(bytearray(arg[1]*dtype.itemsize))
          ul[i] = [lbuf.cast(dtype.fmt)] * warp_size
        elif uop is UOps.DEFINE_VAR:
          ul[i] = [pvals.pop(0)] * warp_size
        elif uop is UOps.SPECIAL:
          if arg[0][0] == 'g': ul[i] = [idxs[2-int(arg[0][-1])]] * warp_size
          elif arg[0][0] == 'l': ul[i] = [x[2-int(arg[0][-1])] for x in warp]
        elif uop is UOps.CONST: ul[i] = [arg] * warp_size
        elif uop is UOps.DEFINE_ACC:
          ul[i] = [[inp[0][0][0]] * warp_size for _ in range(dtype.count)] if dtype.count > 1 else [inp[0][0]] * warp_size
        elif uop is UOps.RANGE:
          if i not in ul: ul[i] = [inp[0][0]] * warp_size
          else:
            for j in range(len(ul[i])):
              ul[i][j] += 1
            if ul[i][0] == inp[1][0]:
              del ul[i]
              i = loop_ends[i] + 1
              continue
        elif uop is UOps.VECTORIZE: ul[i] = inp
        elif uop in {UOps.CAST, UOps.BITCAST}:
          assert dtp[0].fmt and dtype.fmt
          pack_format, unpack_format = str(warp_size) + dtp[0].fmt, str(warp_size) + dtype.fmt
          if uop is UOps.BITCAST: ul[i] = list(struct.unpack(unpack_format, struct.pack(pack_format, *inp[0])))
          else:
            casted = [dtypes.as_const(x, dtype) for x in inp[0]]
            if dtypes.is_int(dtype):
              overflow_adjust = 2**(dtype.itemsize*8 - 1) if not dtypes.is_unsigned(dtype) else 0
              casted = [((x + overflow_adjust) % 2**(dtype.itemsize*8) - overflow_adjust) for x in casted]
            elif dtypes.is_float(dtype):
              casted = [truncate.get(dtype, lambda dt: dt)(x) for x in casted]
            ul[i] = list(struct.unpack(unpack_format, struct.pack(unpack_format, *casted)))
        elif uop is UOps.LOAD:
          if isinstance(dtp[0], ImageDType):
            assert dtype.count == 4
            ul[i] = []
            for j in range(dtype.count):
              ret = []
              for m,ox,oy in zip(inp[0], inp[1][0], inp[1][1]):
                if ox < 0 or ox >= dtp[0].shape[1] or oy < 0 or oy >= dtp[0].shape[0]: ret.append(0)
                else: ret.append(_load(m, ox*4 + oy*dtp[0].shape[1]*4 + j))
              ul[i].append(ret)
          elif dtype.count > 1:
            ul[i] = [load([inp[i][j] if dtp[i].count > 1 else inp[i] for i in range(len(inp))], j) for j in range(dtype.count)]
          else:
            ul[i] = load(inp)
        elif uop is UOps.ASSIGN:
          for j in range(len(inp[0])): inp[0][j] = inp[1][j]
          ul[i] = inp[0]
        elif uop is UOps.GEP:
          assert len(arg) == 1
          ul[i] = inp[0][arg[0]]
        elif uop is UOps.WMMA:
          ul[i] = tcl.wmma_model(arg[4])
        elif uop is UOps.ALU:
          assert all_same([len(x) for x in inp]), f"{[len(x) for x in inp]} doesn't match on {arg}"
          assert all_same([dtype] + dtp) or arg in {BinaryOps.CMPNE, BinaryOps.CMPLT, TernaryOps.WHERE}, f"dtype mismatch on {arg}"
          ul[i] = [exec_alu(arg, dtype, p) for p in zip(*inp)]
        assert i in ul, (uop, dtype, idp, arg)
        i += 1
    return time.perf_counter() - st

class PythonRenderer(Renderer):
  device = "PYTHON"
  def __init__(self):
    if getenv("EMULATE_METAL"): self.device, self.tensor_cores = "METAL", MetalRenderer.tensor_cores
    if getenv("EMULATE_AMD"): self.device, self.tensor_cores = "AMD", AMDRenderer.tensor_cores
    if getenv("EMULATE_CUDA"): self.device, self.tensor_cores = "CUDA", CUDARenderer.tensor_cores
    if getenv("EMULATE_INTEL"): self.device, self.suffix, self.tensor_cores = "INTEL", "INTEL", IntelRenderer.tensor_cores
    if getenv("EMULATE_AMX"): self.device, self.tensor_cores = "CLANG", ClangRenderer.tensor_cores

  def render(self, name:str, uops:List[UOp]) -> str:
    lops = [(u.op, u.dtype, [uops.index(v) for v in u.src], u.arg) for u in uops]
    return base64.b64encode(pickle.dumps(lops)).decode()

class PythonCompiler(Compiler):
  def compile(self, src:str) -> bytes: return base64.b64decode(src)

class PythonAllocator(Allocator):
  def _alloc(self, size, options): return memoryview(bytearray(size))
  def copyin(self, dest, src:memoryview): dest[:] = src
  def copyout(self, dest:memoryview, src): dest[:] = src

class PythonDevice(Compiled):
  def __init__(self, device:str):
    super().__init__(device, PythonAllocator(), PythonRenderer(), PythonCompiler(), PythonProgram)
