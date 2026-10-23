#include "musa.h"
#include <dlfcn.h>
#include <stdbool.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
// #include <stdatomic.h>

// Raises a Python exception and returns false if code is not MUSA_SUCCESS.
static bool gpuAssert(MUresult code, const char *file, int line) {
  if (code == MUSA_SUCCESS)
    return true;

  const char *prefix = "Triton Error [MUSA]: ";
  const char *str;
  muGetErrorString(code, &str);
  char err[1024] = {0};
  strcat(err, prefix);
  strcat(err, str);
  PyGILState_STATE gil_state;
  gil_state = PyGILState_Ensure();
  PyErr_SetString(PyExc_RuntimeError, err);
  PyGILState_Release(gil_state);
  return false;
}

// To be used only *outside* a Py_{BEGIN,END}_ALLOW_THREADS block.
#define MUSA_CHECK_AND_RETURN_NULL(ans)                                        \
  do {                                                                         \
    if (!gpuAssert((ans), __FILE__, __LINE__))                                 \
      return NULL;                                                             \
  } while (0)

// To be used inside a Py_{BEGIN,END}_ALLOW_THREADS block.
#define MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(ans)                          \
  do {                                                                         \
    if (!gpuAssert((ans), __FILE__, __LINE__)) {                               \
      PyEval_RestoreThread(_save);                                             \
      return NULL;                                                             \
    }                                                                          \
  } while (0)

// Used to check if functions exist in old MUSA driver versions.
#define INITIALIZE_FUNCTION_POINTER_IF_NULL(funcPointer, initializerFunction)  \
  do {                                                                         \
    if ((funcPointer) == NULL) {                                               \
      (funcPointer) = (initializerFunction)();                                 \
      if ((funcPointer) == NULL) {                                             \
        return NULL;                                                           \
      }                                                                        \
    }                                                                          \
  } while (0)

static PyObject *getDeviceProperties(PyObject *self, PyObject *args) {
  int device_id;
  if (!PyArg_ParseTuple(args, "i", &device_id))
    return NULL;
  // Get device handle
  MUdevice device;
  muDeviceGet(&device, device_id);

  // create a struct to hold device properties
  int max_shared_mem;
  int max_num_regs;
  int multiprocessor_count;
  int warp_size;
  int sm_clock_rate;
  int mem_clock_rate;
  int mem_bus_width;
  MUSA_CHECK_AND_RETURN_NULL(muDeviceGetAttribute(
      &max_shared_mem, MU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
      device));
  MUSA_CHECK_AND_RETURN_NULL(muDeviceGetAttribute(
      &max_num_regs, MU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, device));
  MUSA_CHECK_AND_RETURN_NULL(muDeviceGetAttribute(
      &multiprocessor_count, MU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
  MUSA_CHECK_AND_RETURN_NULL(
      muDeviceGetAttribute(&warp_size, MU_DEVICE_ATTRIBUTE_WARP_SIZE, device));
  MUSA_CHECK_AND_RETURN_NULL(muDeviceGetAttribute(
      &sm_clock_rate, MU_DEVICE_ATTRIBUTE_CLOCK_RATE, device));
  MUSA_CHECK_AND_RETURN_NULL(muDeviceGetAttribute(
      &mem_clock_rate, MU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device));
  MUSA_CHECK_AND_RETURN_NULL(muDeviceGetAttribute(
      &mem_bus_width, MU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, device));

  return Py_BuildValue("{s:i, s:i, s:i, s:i, s:i, s:i, s:i}", "max_shared_mem",
                       max_shared_mem, "max_num_regs", max_num_regs,
                       "multiprocessor_count", multiprocessor_count, "warpSize",
                       warp_size, "sm_clock_rate", sm_clock_rate,
                       "mem_clock_rate", mem_clock_rate, "mem_bus_width",
                       mem_bus_width);
}

static PyObject *loadBinary(PyObject *self, PyObject *args) {
  const char *name;
  const char *data;
  Py_ssize_t data_size;
  int shared;
  int device;
  if (!PyArg_ParseTuple(args, "ss#ii", &name, &data, &data_size, &shared,
                        &device)) {
    return NULL;
  }
  MUfunction fun;
  MUmodule mod;
  int32_t n_regs = 0;
  int32_t n_spills = 0;
  // create driver handles
  MUcontext pctx = 0;

  Py_BEGIN_ALLOW_THREADS;
  MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(muCtxGetCurrent(&pctx));
  if (!pctx) {
    MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
        muDevicePrimaryCtxRetain(&pctx, device));
    MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(muCtxSetCurrent(pctx));
  }

  MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(muModuleLoadData(&mod, data));
  // MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(muModuleLoad(&mod, data));
  MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
      muModuleGetFunction(&fun, mod, name));
  // get allocated registers and spilled registers from the function
  MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
      muFuncGetAttribute(&n_regs, MU_FUNC_ATTRIBUTE_NUM_REGS, fun));
  MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
      muFuncGetAttribute(&n_spills, MU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fun));
  n_spills /= 4;
  // set dynamic shared memory if necessary
  int shared_optin;
  MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(muDeviceGetAttribute(
      &shared_optin, MU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
      device));
  // supported based on QY2, PH1 is ok here
  // if (shared > 73728 && shared_optin > 73728) {
  //   // MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
  //   //     muFuncSetCacheConfig(fun, MU_FUNC_CACHE_PREFER_SHARED));
  //   int shared_total, shared_static;
  //   MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(muDeviceGetAttribute(
  //       &shared_total, MU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
  //       device));
  //   MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(muFuncGetAttribute(
  //       &shared_static, MU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fun));
  //   MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
  //       muFuncSetAttribute(fun, MU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
  //                          shared_optin - shared_static));
  // }
  Py_END_ALLOW_THREADS;

  if (PyErr_Occurred()) {
    return NULL;
  }
  return Py_BuildValue("(KKii)", (uint64_t)mod, (uint64_t)fun, n_regs,
                       n_spills);
}

// typedef MUresult (*muOccupancyMaxActiveClusters_t)(
//     int *numClusters, MUfunction func, const MUlaunchConfig *config);

// typedef MUresult (*muTensorMapEncodeTiled_t)(
//     MUtensorMap *tensorMap, MUtensorMapDataType tensorDataType,
//     muuint32_t tensorRank, void *globalAddress, const muuint64_t *globalDim,
//     const muuint64_t *globalStrides, const muuint32_t *boxDim,
//     const muuint32_t *elementStrides, MUtensorMapInterleave interleave,
//     MUtensorMapSwizzle swizzle, MUtensorMapL2promotion l2Promotion,
//     MUtensorMapFloatOOBfill oobFill);

// #define defineGetFunctionHandle(name, symbolName)                              \
//   static symbolName##_t name() {                                               \
//     /* Open the shared library */                                              \
//     void *libHandle = dlopen("libmusa.so.1", RTLD_LAZY);                       \
//     if (!libHandle) {                                                          \
//       PyErr_SetString(PyExc_RuntimeError, "Failed to open libmusa.so.1");      \
//       return NULL;                                                             \
//     }                                                                          \
//     /* Clear any existing error */                                             \
//     dlerror();                                                                 \
//     symbolName##_t funcHandle = (symbolName##_t)dlsym(libHandle, #symbolName); \
//     /* Check for errors */                                                     \
//     const char *err = dlerror();                                               \
//     if (err) {                                                                 \
//       PyErr_SetString(PyExc_RuntimeError,                                      \
//                       "Failed to retrieve " #symbolName " from libmusa.so.1"); \
//       dlclose(libHandle);                                                      \
//       return NULL;                                                             \
//     }                                                                          \
//     return funcHandle;                                                         \
//   }

// defineGetFunctionHandle(getMuOccupancyMaxActiveClustersHandle,
//                         muOccupancyMaxActiveClusters);

// defineGetFunctionHandle(getMuTensorMapEncodeTiledHandle,
//                         muTensorMapEncodeTiled);

// static PyObject *occupancyMaxActiveClusters(PyObject *self, PyObject *args) {
//   int clusterDimX = -1, clusterDimY = -1, clusterDimZ = -1,
//       maxActiveClusters = -1;
//   int shared = 0;
//   MUfunction func;

//   if (!PyArg_ParseTuple(args, "Kiiii", &func, &shared, &clusterDimX,
//                         &clusterDimY, &clusterDimZ)) {
//     return NULL;
//   }

//   // Let each SM have one block
//   int maxActiveBlocks = 1;
//   Py_BEGIN_ALLOW_THREADS;
//   MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(muFuncSetAttribute(
//       func, MU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared));
//   Py_END_ALLOW_THREADS;

//   MUlaunchAttribute launchAttr[1];
//   launchAttr[0].id = MU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
//   launchAttr[0].value.clusterDim.x = clusterDimX;
//   launchAttr[0].value.clusterDim.y = clusterDimY;
//   launchAttr[0].value.clusterDim.z = clusterDimZ;
//   MUlaunchConfig config;
//   config.gridDimX = clusterDimX;
//   config.gridDimY = maxActiveBlocks * clusterDimY;
//   config.gridDimZ = clusterDimZ;
//   config.blockDimX = 128;
//   config.blockDimY = 1;
//   config.blockDimZ = 1;
//   config.sharedMemBytes = shared;
//   config.hStream = 0;
//   config.numAttrs = 1;
//   config.attrs = launchAttr;

//   static muOccupancyMaxActiveClusters_t muOccupancyMaxActiveClusters = NULL;
//   INITIALIZE_FUNCTION_POINTER_IF_NULL(muOccupancyMaxActiveClusters,
//                                       getMuOccupancyMaxActiveClustersHandle);

//   Py_BEGIN_ALLOW_THREADS;
//   MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(muFuncSetAttribute(
//       func, MU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1));
//   MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
//       muOccupancyMaxActiveClusters(&maxActiveClusters, func, &config));
//   Py_END_ALLOW_THREADS;
//   return PyLong_FromLong(maxActiveClusters);
// }

// static PyObject *setPrintfFifoSize(PyObject *self, PyObject *args) {
//   long size;
//   if (!PyArg_ParseTuple(args, "l", &size)) {
//     return NULL;
//   }
//   if (size < 0) {
//     PyErr_SetString(PyExc_ValueError, "fifo size must be non-negative");
//     return NULL;
//   }

//   Py_BEGIN_ALLOW_THREADS;

//   // Ensure we have an active context.
//   MUcontext ctx = NULL;
//   MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(muCtxGetCurrent(&ctx));
//   if (!ctx) {
//     MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
//         muDevicePrimaryCtxRetain(&ctx, /*device=*/0));
//     MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(muCtxSetCurrent(ctx));
//   }

//   // We can't set the fifo size after running a kernel that calls printf.  This
//   // is true even if the set() call is a nop and the new size is the same as the
//   // old size.
//   //
//   // This is unfriendly, so check if the old size matches the new size, and skip
//   // the set() call if so.
//   size_t oldSize = 0;
//   MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
//       muCtxGetLimit(&oldSize, MU_LIMIT_PRINTF_FIFO_SIZE));
//   if (oldSize != size) {
//     MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
//         muCtxSetLimit(MU_LIMIT_PRINTF_FIFO_SIZE, size));
//   }

//   Py_END_ALLOW_THREADS;
//   return Py_None;
// }

// Simple helper to experiment creating TMA descriptors on the host.
// This is a useful to test TMA operations independently.
static PyObject *fill1DTMEDescriptor(PyObject *self, PyObject *args) {
  unsigned long long global_address;
  uint64_t dim;
  uint32_t tensorDim;
  int elementSize;
  unsigned long long desc_address;
  if (!PyArg_ParseTuple(args, "KKiiK", &global_address, &dim, &tensorDim,
                        &elementSize, &desc_address)) {
    return NULL;
  }
  uint64_t dims[1] = {dim};
  uint64_t globalStrides[1] = {dim * elementSize};
  uint32_t elementStrides[1] = {1};
  MUtensorDescriptorDataType type;
  // FIXME: mcj. shall we specify specific type, like u8/s8, fp16/bf16, fp32, tf32?
  switch (elementSize) {
  case 1:
    type = MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT8;
    break;
  case 2:
    type = MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT16;
    break;
  case 4:
    type = MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT32;
    break;
  default:
    PyErr_SetString(PyExc_ValueError, "elementSize must be 1, 2, or 4");
  }
  assert((elementSize * tensorDim) >= 32 && "block size too small.");
  int rank = 1;

  MUresult result = muTensorDescriptorEncode(
      (MUtensorDescriptor *)desc_address, type, rank, (void *)global_address,
      dims, globalStrides, MU_TENSOR_DESCRIPTOR_INTERLEAVE_NONE, 0);
  assert(result == MUSA_SUCCESS);
  return Py_None;
}

// Simple helper to experiment creating TME descriptors on the host.
// This is a useful to test TMA operations independently.
static PyObject *fill2DTMEDescriptor(PyObject *self, PyObject *args) {
  unsigned long long global_address;
  uint64_t dims[2];
  uint32_t tensorDims[2];
  int elementSize;
  unsigned long long desc_address;
  if (!PyArg_ParseTuple(args, "KKKiiiK", &global_address, &dims[1], &dims[0],
                        &tensorDims[1], &tensorDims[0], &elementSize,
                        &desc_address)) {
    return NULL;
  }
  uint64_t globalStrides[2] = {dims[0] * elementSize,
                               dims[0] * dims[1] * elementSize};
  uint32_t elementStrides[2] = {1, 1};
  MUtensorDescriptorDataType type;
  switch (elementSize) {
  case 1:
    type = MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT8;
    break;
  case 2:
    type = MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT16;
    break;
  case 4:
    type = MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT32;
    break;
  default:
    PyErr_SetString(PyExc_ValueError, "elementSize must be 1, 2, or 4");
  }
  int rank = 2;

  MUresult result = muTensorDescriptorEncode(
      (MUtensorDescriptor *)desc_address, type, rank, (void *)global_address,
      dims, globalStrides, MU_TENSOR_DESCRIPTOR_INTERLEAVE_NONE, 0);
  assert(result == MUSA_SUCCESS);
  Py_INCREF(Py_None);
  return Py_None;
}

static PyObject *fill3DTMEDescriptor(PyObject *self, PyObject *args) {
  unsigned long long global_address;
  uint64_t dims[3];
  uint32_t tensorDims[3];
  int elementSize;
  unsigned long long desc_address;
  if (!PyArg_ParseTuple(args, "KKKKiiiiK", &global_address, &dims[2], &dims[1],
                        &dims[0], &tensorDims[2], &tensorDims[1],
                        &tensorDims[0], &elementSize, &desc_address)) {
    return NULL;
  }
  uint64_t globalStrides[3] = {dims[0] * elementSize,
                               dims[0] * dims[1] * elementSize,
                               dims[0] * dims[1] * dims[2] * elementSize};
  uint32_t elementStrides[3] = {1, 1, 1};
  MUtensorDescriptorDataType type;
  switch (elementSize) {
  case 1:
    type = MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT8;
    break;
  case 2:
    type = MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT16;
    break;
  case 4:
    type = MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT32;
    break;
  default:
    PyErr_SetString(PyExc_ValueError, "elementSize must be 1, 2, or 4");
  }
  int rank = 3;

  MUresult result = muTensorDescriptorEncode(
      (MUtensorDescriptor *)desc_address, type, rank, (void *)global_address, dims,
      globalStrides, MU_TENSOR_DESCRIPTOR_INTERLEAVE_NONE, 0);
  assert(result == MUSA_SUCCESS);
  Py_INCREF(Py_None);
  return Py_None;
}

static PyMethodDef ModuleMethods[] = {
    {"load_binary", loadBinary, METH_VARARGS,
     "Load provided mubin into MUSA driver"},
    {"get_device_properties", getDeviceProperties, METH_VARARGS,
     "Get the properties for a given device"},
    // TODO(lingfeng.qiu): Support functions below.
    // {"muOccupancyMaxActiveClusters", occupancyMaxActiveClusters,
    // METH_VARARGS,
    //  "Python interface for muOccupancyMaxActiveClusters function"},
    // {"set_printf_fifo_size", setPrintfFifoSize, METH_VARARGS,
    //  "Python interface for muCtxSetLimit(MU_LIMIT_PRINTF_FIFO_SIZE, x), which
    //  " "controls how many bytes can be streamed from kernels before data
    //  starts " "being dropped.  This inherits all the limitations of this
    //  call; in " "particular it's an error to change this value after
    //  launching any kernel " "that calls printf()."},
    {"fill_1d_tma_descriptor", fill1DTMEDescriptor, METH_VARARGS,
     "create a tme 1D descriptor"},
    {"fill_2d_tma_descriptor", fill2DTMEDescriptor, METH_VARARGS,
     "create a tme 2D descriptor"},
    {"fill_3d_tma_descriptor", fill3DTMEDescriptor, METH_VARARGS,
     "create a tme 3D descriptor"},

    {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {PyModuleDef_HEAD_INIT, "musa_utils",
                                       NULL, // documentation
                                       -1,   // size
                                       ModuleMethods};

PyMODINIT_FUNC PyInit_musa_utils(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if (m == NULL) {
    return NULL;
  }

  PyModule_AddFunctions(m, ModuleMethods);

  return m;
}
