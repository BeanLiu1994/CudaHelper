make cuda easier to use.

# cudaInitializer usage

initial:   cudaInitializer::Init();

check:     cudaInitializer::CudaOK();


calls cudacudaDeviceReset() at program exit.

# CuPtr usage

CuPtr:

* Constructor
  
* CuGetResult

* GetDevicePtr

* CuSyncDevice  (static)

these four functions should be enough for simple cuda program.

CuPtr_Const use a const ptr for initialization, and can not copy data from device to host.

throws runtime_error if err;

test 1
```
f(T *a, ...)
{
  CuPtr<T> da(a, size);//  malloc: size*sizeof(T) , and copy to device.
  ...
  kernel<<<?,?>>>(
                da.GetDevicePtr(),
                ...
                );
  da.CuGetResult();//  sync and copy to host.
}
```

test 2
```
f(T *a, ...)
{
  CuPtr<T> da(nullptr, size*sizeof(T));//  malloc: size*sizeof(T) , copy nothing.
  ...
  kernel<<<?,?>>>(
                da.GetDevicePtr(),
                ...
                );
  da.CuGetResult(a);//  sync and copy to host.
}
```
