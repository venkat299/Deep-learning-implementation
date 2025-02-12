
from typing import Callable
from typing import List
from numpy import ndarray

Array_Function = Callable[[ndarray], ndarray]
Chain = List[Array_Function]

# derivative
def deriv(func: Array_Function, 
          input_:ndarray, 
          delta:float=0.001) -> ndarray:
    return (func(input_+delta)-func(input_-delta))/(2*delta)




# chain function
def chain_len_2(chain:Chain,
                a: ndarray)->ndarray:
      assert len(chain)==2,\
      "Length of input 'chain' should be 2"
      
      f1=chain[0]
      f2=chain[1]
      return f2(f1(a))
