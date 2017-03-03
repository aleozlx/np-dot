import numpy as np
import time, traceback

MEM_LIMIT = 2 <<30

mem = lambda shape, dtype: np.product(shape) * np.dtype(dtype).itemsize
dims = map(lambda x:1<<x, xrange(6,20,2))

def test01(dtype):
    print('+ %s matrix multiplication'%dtype)
    shapes = [(d,d) for d in dims]
    for shape in filter(lambda i: mem(i, dtype) < MEM_LIMIT/2, shapes):
        m1 = np.random.rand(*shape).astype(dtype)
        m2 = np.random.rand(*shape).astype(dtype)
        start = time.time()
        ret = np.dot(m1,m2)
        end = time.time()
        print(shape, round(end - start, 4))
        
def test02(dtype):
    print('+ %s 3d vectors dot product'%dtype)
    shapes = [(d,3) for d in dims]
    for shape in filter(lambda i: mem(i, dtype) < MEM_LIMIT/2, shapes):
        m1 = np.random.rand(*shape).astype(dtype)
        m2 = np.random.rand(*shape).astype(dtype)
        start = time.time()
        ret = np.sum(m1*m2, axis=1)
        end = time.time()
        print(shape, round(end - start, 4))
        
def test03(dtype):
    print('+ %s 20d vectors dot product'%dtype)
    shapes = [(d,20) for d in dims]
    for shape in filter(lambda i: mem(i, dtype) < MEM_LIMIT/2, shapes):
        m1 = np.random.rand(*shape).astype(dtype)
        m2 = np.random.rand(*shape).astype(dtype)
        start = time.time()
        ret = np.sum(m1*m2, axis=1)
        end = time.time()
        print(shape, round(end - start, 4))
        
def test04(dtype):
    print('+ %s 100d vectors dot product'%dtype)
    shapes = [(d,100) for d in dims]
    for shape in filter(lambda i: mem(i, dtype) < MEM_LIMIT/2, shapes):
        m1 = np.random.rand(*shape).astype(dtype)
        m2 = np.random.rand(*shape).astype(dtype)
        start = time.time()
        ret = np.sum(m1*m2, axis=1)
        end = time.time()
        print(shape, round(end - start, 4))

def test05(dtype):
    print('+ %s 1000d vectors dot product'%dtype)
    shapes = [(d,1000) for d in dims]
    for shape in filter(lambda i: mem(i, dtype) < MEM_LIMIT/2, shapes):
        m1 = np.random.rand(*shape).astype(dtype)
        m2 = np.random.rand(*shape).astype(dtype)
        start = time.time()
        ret = np.sum(m1*m2, axis=1)
        end = time.time()
        print(shape, round(end - start, 4))
        
def test06(dtype):
    print('+ %s tensors einsum(ijk,ijk->ij)'%dtype)
    shapes = [(d,d,d) for d in dims]
    for shape in filter(lambda i: mem(i, dtype) < MEM_LIMIT/2, shapes):
        m1 = np.random.rand(*shape).astype(dtype)
        m2 = np.random.rand(*shape).astype(dtype)
        start = time.time()
        ret = np.einsum('ijk,ijk->ij', m1,m2)
        end = time.time()
        print(shape, round(end - start, 4))
        
def test07(dtype):
    print('+ %s tensors einsum(ijs,ijt->st)'%dtype)
    shapes = [(d,d,d) for d in dims]
    for shape in filter(lambda i: mem(i, dtype) < MEM_LIMIT/2, shapes):
        m1 = np.random.rand(*shape).astype(dtype)
        m2 = np.random.rand(*shape).astype(dtype)
        start = time.time()
        ret = np.einsum('ijs,ijt->st', m1,m2)
        end = time.time()
        print(shape, round(end - start, 4))
        
if __name__ == '__main__':
    for func_name, test in sorted((name,obj) for name,obj in globals().items()
                 if name.startswith('test') and hasattr(obj, '__call__')):
        test('float64')
        test('float32')
