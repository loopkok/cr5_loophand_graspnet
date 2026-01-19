import numpy as np

class AdmittanceController():
    """导纳控制器"""
    def __init__(self,_mass,_stiffness,_damping,dt):
        self._mass = _mass
        self._stiffness = _stiffness
        self._damping = _damping
        self.dt = dt
        self.x_error = np.zeros(3)
        self.vel_error = np.zeros(3)
        self.Force_constant = np.zeros(3)
    
    def update(self, Force_ext:np.ndarray):
        """计算柔顺所需要的位移增量"""
        acc = (Force_ext - self.Force_constant - self._damping*self.vel_error - self._stiffness*self.x_error) / self._mass
        self.vel_error += acc * self.dt
        self.x_error += self.vel_error * self.dt
        return self.x_error.copy()
    
    def set_Force_constant(self,F):
        """设置外部接触的恒力"""
        self.Force_constant = F

    def reset(self):
        """重置导纳控制器"""
        self.vel_error = np.zeros(3)
        self.x_error = np.zeros(3)