# Existing Solutions for Unified Array/Matrix Backends

## Summary

After extensive research, here are the existing solutions for unified array/matrix operations with automatic hardware acceleration detection:

---

## **Direct Competitors (Most Similar)**

### 1. **Accelera** ⭐
- **What it does**: Python library that automatically detects hardware accelerators and provides unified interface for dense/sparse matrix operations
- **Backends**: NumPy, PyTorch
- **Features**: 
  - Automatic hardware detection
  - Unified interface for NumPy arrays and PyTorch tensors
  - Supports dense and sparse matrices
  - Automatic memory management
  - Customizable chunking strategies
- **Status**: Available on PyPI
- **Limitations**: 
  - Only supports NumPy and PyTorch (no JAX)
  - Unknown maturity/community size
  - May not have sparse matrix support fully developed

### 2. **ArrayFire**
- **What it does**: General-purpose GPU library with unified interface
- **Backends**: CUDA, OpenCL, CPU
- **Features**:
  - High-level interface for matrix operations
  - Multiple backend support
  - Cross-platform (CUDA, OpenCL)
  - Dense and sparse linear algebra
- **Limitations**:
  - C++ library with Python bindings (more complex)
  - Primarily GPU-focused
  - Less Python-native feel

### 3. **KeOps**
- **What it does**: Efficient kernel operations on GPU with automatic differentiation
- **Backends**: NumPy, PyTorch, JAX
- **Features**:
  - Symbolic tensor interface
  - Handles large datasets (memory efficient)
  - Automatic differentiation
  - Kernel matrix-vector products
- **Limitations**:
  - Specialized for kernel operations
  - Not general-purpose matrix operations
  - Optimized for specific use cases (kernel methods, point clouds)

---

## **Individual Libraries (Single Backend)**

These are powerful but require manual backend selection:

### 4. **CuPy**
- NumPy-compatible interface for NVIDIA GPUs
- Automatic GPU detection
- **Limitation**: Only CUDA/NVIDIA, not unified with CPU/other backends

### 5. **JAX**
- NumPy-like API with automatic GPU/TPU
- Automatic hardware detection
- **Limitation**: JAX-specific, doesn't unify with PyTorch/NumPy transparently

### 6. **PyTorch**
- Automatic GPU detection
- Tensor operations
- **Limitation**: PyTorch-specific, doesn't abstract backend selection

### 7. **NumPy**
- CPU-optimized with BLAS/LAPACK integration
- Can link to MKL, OpenBLAS for hardware optimization
- **Limitation**: CPU-only, no GPU abstraction

---

## **Specialized Libraries**

### 8. **nvmath-python** (NVIDIA)
- GPU-accelerated math functions
- **Limitation**: NVIDIA-specific, not general-purpose

### 9. **Numba**
- JIT compiler for GPU acceleration
- **Limitation**: Requires code compilation, not a unified interface

### 10. **cuSPARSELt** (NVIDIA)
- Sparse matrix operations on CUDA
- **Limitation**: CUDA/NVIDIA-specific, sparse-only

---

## **Comparison: Your Module vs. Existing Solutions**

| Feature | Your Module | Accelera | ArrayFire | KeOps | Individual Libs |
|---------|-------------|----------|-----------|-------|------------------|
| **Auto backend selection** | ✅ Yes | ✅ Yes | ✅ Yes | ❌ No | ❌ No |
| **Unified NumPy/JAX/PyTorch** | ✅ Yes | ❌ NumPy+PyTorch only | ❌ C++ backends | ❌ Specialized | ❌ Single backend |
| **Cross-platform (macOS/Windows/Linux)** | ✅ Yes | ✅ Likely | ✅ Yes | ✅ Yes | ✅ Varies |
| **Sparse matrix support** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Limited | ✅ Varies |
| **OS detection** | ✅ Yes | ❓ Unknown | ❓ Unknown | ❌ No | ❌ No |
| **JAX support** | ✅ Yes | ❌ No | ❌ No | ✅ Yes | ✅ JAX only |
| **MPS (Apple Silicon) support** | ✅ Yes | ❓ Unknown | ❌ No | ❌ No | ✅ PyTorch only |
| **TPU support** | ✅ Yes | ❌ No | ❌ No | ❌ No | ✅ JAX only |
| **Python-native API** | ✅ Yes | ✅ Yes | ⚠️ C++ bindings | ✅ Yes | ✅ Yes |
| **Maturity/Community** | ⚠️ New | ❓ Unknown | ✅ Established | ✅ Active | ✅ Very established |

---

## **Key Findings**

### **What Exists:**
1. **Accelera** appears to be the closest match - provides unified backend selection for NumPy/PyTorch
2. **ArrayFire** provides unified interface but at C++ level with Python bindings
3. **Individual libraries** are powerful but require manual backend management

### **What's Missing:**
1. **No mature library** that unifies NumPy + JAX + PyTorch with auto-selection
2. **OS-aware backend selection** (most don't explicitly handle macOS MPS, Windows CUDA, Linux CUDA differently)
3. **TPU support** in unified interfaces is rare (mostly JAX-only)
4. **Python-native unified sparse matrix** abstraction across backends

### **Your Module's Unique Value:**
1. ✅ **Truly unified** across NumPy/JAX/PyTorch
2. ✅ **OS-aware** (handles macOS MPS, Windows CUDA, Linux CUDA)
3. ✅ **TPU support** via JAX
4. ✅ **Python-native** design
5. ✅ **Specific domain** (grid neighborhoods) built-in
6. ✅ **Automatic prioritization** (JAX TPU → JAX GPU → PyTorch CUDA → PyTorch MPS → NumPy)

---

## **Recommendations**

### **Option 1: Use Your Module**
**Pros:**
- Meets your specific requirements
- Full control over API and features
- Better suited for your grid counting domain
- Unified across all major backends

**Cons:**
- New codebase to maintain
- Smaller community than established libraries

### **Option 2: Use Accelera (if suitable)**
**Pros:**
- Existing, tested codebase
- Active maintenance (if community exists)
- NumPy + PyTorch support

**Cons:**
- No JAX support
- Unknown OS-aware features
- May not fit your sparse matrix needs
- Need to evaluate if it actually does what you need

### **Option 3: Hybrid Approach**
- Use your module for JAX/MPS/TPU support
- Reference Accelera patterns for NumPy/PyTorch abstractions
- Consider contributing back unified features

---

## **Action Items**

1. **Evaluate Accelera** more closely:
   - Check GitHub repository and documentation
   - Test if it meets your requirements
   - Compare API design to your module

2. **Consider Contributing**:
   - If Accelera is close, contribute JAX support
   - Or contribute your unified approach as an enhancement

3. **Keep Your Module** if:
   - Accelera doesn't meet requirements
   - You need JAX/TPU/MPS support
   - Your specific domain features (Manhattan neighborhoods) are important

---

## **References**

- **Accelera**: https://pypi.org/project/accelera/
- **ArrayFire**: https://github.com/arrayfire/arrayfire-python
- **KeOps**: https://www.kernel-operations.io/keops/
- **CuPy**: https://cupy.dev/
- **JAX**: https://github.com/google/jax
- **PyTorch**: https://pytorch.org/

---

## **Conclusion**

**Your module fills a real gap**: While several libraries provide unified interfaces, none provide a truly unified NumPy+JAX+PyTorch abstraction with automatic OS-aware backend selection. Accelera is the closest, but lacks JAX and potentially OS-specific features.

Your module is valuable if you need:
- Cross-backend unification (NumPy/JAX/PyTorch)
- OS-aware hardware detection
- Specific domain features (grid neighborhoods)
- Full control over API design

Consider evaluating Accelera first, but your module appears to solve a legitimate problem that existing solutions don't fully address.

