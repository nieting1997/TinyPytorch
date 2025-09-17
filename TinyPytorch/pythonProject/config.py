import contextlib

class Config:
    enable_backprop = True

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name) # 预处理
    setattr(Config, name, value)
    try:
        yield # 当进入 with 语句块时，程序会执行到 yield 暂停，先去执行 with 语句块里的代码。
    finally: # 后处理
        setattr(Config, name, old_value)

def no_grad():
    return using_config('enable_backprop', False)