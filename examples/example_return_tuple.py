import warp as wp

wp.init()


@wp.func
def plus_minus(a: wp.vec3, b: wp.vec3):
    return a + b, a - b


@wp.func
def test_plus(a: wp.vec3, b: wp.vec3):
    return a + b


@wp.kernel
def mykernel():
    a = wp.vec3(0.0, 0.0, 0.0)
    b = wp.vec3(1.0, 1.0, 1.0)
    c, d = plus_minus(a, b)
    print(c)
    print(d)
    x = test_plus(a, b)
    print(x)


wp.launch(mykernel, dim=1, inputs=[], device="cuda")
wp.synchronize()
