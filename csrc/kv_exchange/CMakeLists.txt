# MF+AIV kv_exchange: MLA HiCache L2<->L1 IO over registered MemFabric host
# blocks. Legacy Host and Kernel are wired in cmake/legacy_kernels.cmake;
# the Torch schema self-registers through torch_adapter.cpp.
sgl_register_operator(
    ID kv_exchange
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE kv_exchange
    TORCH_ADAPTER torch_adapter.cpp
    SUPPORTED_ARCHS 2201 3510
    HOST_SOURCES op_host/kv_exchange.cpp
    KERNEL_SOURCES op_kernel/kv_exchange_kernel.cpp
)
