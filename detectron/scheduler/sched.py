import collections
import weakref
import threading
from abc import ABC, abstractmethod

class ModelWrapper(ABC):
    
    @abstractmethod
    def release(self):
        pass

class LRUModelScheduler:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(LRUModelScheduler, cls).__new__(cls)
                cls._instance._init(*args, **kwargs)
        return cls._instance

    def _init(self, capacity=5):
        self.capacity = capacity
        self.cache = collections.OrderedDict()
    
    def get_model(self, key):
        """获取模型，如果模型存在则更新其使用状态"""
        if key not in self.cache:
            return None
        # 将模型移到末尾，表示最近使用
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put_model(self, key, model:ModelWrapper):
        """添加/更新模型到调度器"""
        if key in self.cache:
            # 如果模型已存在，则更新
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.capacity:
            # 如果超出容量，则删除最少使用的模型
            oldest_key, oldest_model = self.cache.popitem(last=False)
            self._destroy_model(oldest_model)
        
        # 添加新模型
        self.cache[key] = weakref.ref(model, self._model_finalizer)
    
    def _destroy_model(self, model:ModelWrapper):
        """销毁模型（释放资源）"""
        if model is not None:
            model.release()
    
    def _model_finalizer(self, weak_ref):
        """模型被销毁时的回调"""
        model = weak_ref()
        if model is not None:
            self._destroy_model(model)
    
    def clear(self):
        """清空缓存并销毁所有模型"""
        for model in self.cache.values():
            self._destroy_model(model)
        self.cache.clear()
