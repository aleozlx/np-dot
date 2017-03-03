from __future__ import print_function
import numpy as np
import time, traceback

MEM_LIMIT = 2 <<30

mem = lambda shape, dtype: np.product(shape) * np.dtype(dtype).itemsize
dims = map(lambda x:1<<x, xrange(6,20,2))

def timeit(func):
    start = time.time()
    func()
    end = time.time()
    return round(end - start, 4)

def rand(shape, dtype, n):
    return tuple(np.random.rand(*shape).astype(dtype) for i in xrange(n))

def test01(dtype):
    print('+ %s matrix multiplication'%dtype)
    shapes = [(d,d) for d in dims]
    for shape in filter(lambda i: mem(i, dtype) < MEM_LIMIT/2, shapes):
        m1, m2 = rand(shape, dtype, 2)
        print(shape, timeit(lambda: np.dot(m1,m2)))
        
def test02(dtype):
    print('+ %s 3d vectors dot product'%dtype)
    shapes = [(d,3) for d in dims]
    for shape in filter(lambda i: mem(i, dtype) < MEM_LIMIT/2, shapes):
        m1, m2 = rand(shape, dtype, 2)
        print(shape, timeit(lambda: np.sum(m1*m2, axis=1)))
        
def test03(dtype):
    print('+ %s 20d vectors dot product'%dtype)
    shapes = [(d,20) for d in dims]
    for shape in filter(lambda i: mem(i, dtype) < MEM_LIMIT/2, shapes):
        m1, m2 = rand(shape, dtype, 2)
        print(shape, timeit(lambda: np.sum(m1*m2, axis=1)))
        
def test04(dtype):
    print('+ %s 100d vectors dot product'%dtype)
    shapes = [(d,100) for d in dims]
    for shape in filter(lambda i: mem(i, dtype) < MEM_LIMIT/2, shapes):
        m1, m2 = rand(shape, dtype, 2)
        print(shape, timeit(lambda: np.sum(m1*m2, axis=1)))

def test05(dtype):
    print('+ %s 1000d vectors dot product'%dtype)
    shapes = [(d,1000) for d in dims]
    for shape in filter(lambda i: mem(i, dtype) < MEM_LIMIT/2, shapes):
        m1, m2 = rand(shape, dtype, 2)
        print(shape, timeit(lambda: np.sum(m1*m2, axis=1)))
        
def test06(dtype):
    print('+ %s tensors einsum(ijk,ijk->ij)'%dtype)
    shapes = [(d,d,d) for d in dims]
    for shape in filter(lambda i: mem(i, dtype) < MEM_LIMIT/2, shapes):
        m1, m2 = rand(shape, dtype, 2)
        print(shape, timeit(lambda: np.einsum('ijk,ijk->ij', m1,m2)))
        
def test07(dtype):
    print('+ %s tensors einsum(ijs,ijt->st)'%dtype)
    shapes = [(d,d,d) for d in dims]
    for shape in filter(lambda i: mem(i, dtype) < MEM_LIMIT/2, shapes):
        m1, m2 = rand(shape, dtype, 2)
        print(shape, timeit(lambda: np.einsum('ijs,ijt->st', m1,m2)))
        
if __name__ == '__main__':
    for func_name, test in sorted((name,obj) for name,obj in globals().items()
                 if name.startswith('test') and hasattr(obj, '__call__')):
        test('float64')
        test('float32')
