#pragma once

#include <cuda_runtime.h>
#include "CudaManager.h"
#include "Culocator.h"
#include <stdexcept>
#include <iostream>
#include <string>
#include <typeinfo>

// 后续希望支持move操作,const的更方便的用法,模板参数的特殊处理等。
template<typename _tyIn> // 元素类型
class CuPtr
{
public:
	typedef typename std::decay<_tyIn>::type _ty;
	typedef typename std::remove_reference<_tyIn>::type _ty_noref;

	static const size_t type_size = sizeof(_ty);
protected:
	size_t elem_size = 0, mem_size = 0;

	_ty_noref * ptr = nullptr;
	void* device_ptr = nullptr;


	void size_assign(size_t _size)
	{
		elem_size = _size;
		mem_size = _size * type_size;

		if (mem_size == 0)
		{
			std::cerr <<
				"You're setting a CuPtr of 0 size, Please check your parameter if it's wrong.  CuPtr<...>(size, value)"
				<< std::endl;
			return;
		}
	}
public:
	explicit CuPtr(size_t _size, _ty_noref* _ptr)
	{
		assign(_size, _ptr);
	}
	explicit CuPtr(size_t _size, _ty initVal = _ty())
	{
		assign(_size, initVal);
	}
	CuPtr() = default;

	template<typename rhsType, typename t = typename std::enable_if<std::is_same<_ty, typename rhsType::_ty>::value>>
	CuPtr(const rhsType&) = delete;

	template<typename rhsType, typename t = typename std::enable_if<std::is_same<_ty, typename CuPtr<rhsType>::_ty>::value>>
	CuPtr(CuPtr<rhsType>&& rhs) noexcept
	{
		//assert(type_size == rhs.type_size);
		using std::swap;
		swap(elem_size, rhs.elem_size);
		swap(mem_size, rhs.mem_size);
		ptr = rhs.ptr;
		device_ptr = rhs.device_ptr;
		rhs.device_ptr = nullptr;
	}

	void free()
	{
		if (device_ptr)
		{
			_cu_free(device_ptr);
			device_ptr = nullptr;
		}
	}

	void assign(size_t _size, _ty initVal = _ty())
	{
		free();
		size_assign(_size);
		_cu_malloc(&device_ptr, mem_size);
		_cu_memset(device_ptr, mem_size, initVal);
	}

	void assign(size_t _size, _ty_noref* _ptr)
	{
		free();
		size_assign(_size);
		ptr = _ptr;

		_cu_malloc(&device_ptr, mem_size);
		if (ptr != nullptr)
			_cu_copyToDevice(device_ptr, mem_size, ptr);
		else
			_cu_memset(device_ptr, mem_size, 0);
	}

	_ty* operator()()
	{
		return static_cast<_ty*>(device_ptr);
	}
	const _ty* operator()() const
	{
		return static_cast<_ty*>(device_ptr);
	}

	void GetResult()
	{
		static_assert(std::is_same<_ty_noref, _ty>::value, "CuPtr<const T> can't call GetResult.");
		if (ptr != nullptr)
		{
			_cu_getResult(device_ptr, mem_size, ptr);
		}
		else
		{
			std::cerr << "nothing send to cpu ptr, cause ptr is null. It's not expected to run here. You may check this code if there is any bug." << std::endl;
		}
	}

	void GetResult(_ty* OutPtr)
	{
		gpuErrchk(cudaGetLastError());
		if (OutPtr != nullptr)
		{
			_cu_getResult(device_ptr, mem_size, OutPtr);
		}
		else
		{
			throw std::runtime_error("null pointer input.");
		}
	}
	~CuPtr()
	{
		free();
		gpuErrchk(cudaGetLastError());// in case there may be something wrong.
	}

	static void SyncDevice()
	{
		_cu_syncDevice();
	}

	constexpr size_t Get_type_size() { return type_size; }
	size_t Get_elem_size() { return elem_size; }
	size_t Get_mem_size() { return mem_size; }
};
