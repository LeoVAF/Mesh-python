import numpy as np
import numpy.typing as npt

class Converter():
  ''' Class to simulate the microgrid DC/DC converter.
  
  Args:
    cost_per_kw (:type:`int | float`): Converter cost per kW of nominal capacity in [$/kW].
    cost_scale (:type:`int | float`): Cost scale factor for converter, where a higher power results in a lower cost per kW in [decimal].
    efficiency (:type:`int | float`): Converter efficiency between 0 and 1.
    lifetime (:type:`int | float`): Converter lifetime in time intervals.
  
  Raises:
    TypeError: If the input is not the expected type.
    ValueError: If the input is not the allowed value.
  '''

  def __init__(self,
               cost_per_kw: int | float,
               cost_scale: int | float = 0.95,
               efficiency: int | float = 0.95,
               lifetime: int | float = 10):

    self.cost_per_kw: int | float
    ''' Converter cost per kW of nominal capacity in [$/kW]. '''
    self.cost_scale: int | float
    ''' Scaling cost factor for converter, where a higher power results in a lower cost per kW in [decimal]. '''
    self.efficiency: int | float
    ''' Converter efficiency between 0 and 1. '''
    self.lifetime: int | float
    ''' Converter lifetime in time intervals. '''
    self.operation_cost: float = 0.0
    ''' Total costs of the converter in the microgrid during the operation simulation in [$]. '''

    self.cost_per_kw = cost_per_kw
    self.cost_scale = cost_scale
    self.efficiency = efficiency
    self.lifetime = lifetime

  def economic_analysis(self,
                        rated_power: int | float,
                        project_lifetime_intervals: npt.NDArray[np.integer],
                        maintenance_cost_rate: int | float,
                        discount_rate: int | float,
                        CRF: int | float) -> float:
    r''' Performs the economic analysis of the converter using the Net Present Cost (NPC) approach.

    The total NPC of the converter is given by:

    .. math::
        NPC_{conv} = \text{IC}_{conv} + \text{NPV}_{om} + \text{NPV}_{repl}.

    Where:
    
    - :math:`\text{IC}_{conv}` is the installation cost;
    - :math:`\text{NPV}_{om}` is the Net Present Value of annual operation and maintenance costs;
    - :math:`\text{NPV}_{repl}` is the Net Present Value of replacement costs during the project lifetime.

    The installation cost is calculated as:

    .. math::
        \text{IC}_{conv} = C_{kw} \cdot P_{rated}^{\tau_{conv}}.

    :math:`C_{kw}` is the cost per kW of nominal capacity for the converter, :math:`P_{rated}` is the rated power of the distributed energy resources and :math:`\tau_{conv}` is the converter economies of scale. The operation and maintenance costs are calculated as:

    .. math::
        \text{NPV}_{om} = \sum^{T}_{t=1}\frac{\text{IC}_{conv} \cdot \tau_{om}}{(1 + d)^t}.

    :math:`T` is the project lifetime in time intervals, :math:`d` is the discount rate per interval (assumed to be constant) in [decimal] and :math:`\tau_{om}` is the operation and maintenance cost rate in [decimal]. The replacement costs occur every :attr:`lifetime` intervals and are equal to the installation cost, discounted to present value according to the following equation:
    
    .. math::
      \text{NPV}_{repl} = \sum^{T}_{t=1}\frac{\left(\left\lfloor \frac{t}{T_{\text{repl}}} \right\rfloor - \left\lfloor \frac{t-1}{T_{\text{repl}}} \right\rfloor\right) \cdot \text{IC}_{conv}}{(1 + d)^t},

    where :math:`T_{repl} = I^{\text{lifetime}}_{\text{conv}}` is the time when the euipament must to be replaced.

    Args:
        rated_power (:type:`int | float`): The power supported by the converter in [kW].
        project_lifetime_intervals (:type:`npt.NDArray[np.integer]`): Intervals of project lifetime.
        maintenance_cost_rate (:type:`int | float`): Operation and maintenance cost rate based on installation costs in [decimal].
        discount_rate (:type:`int | float`): Discount rate (per interval) during the project lifetime in [decimal].
        CRF (:type:`int | float`): Capital Recovery Factor (CRF) during the project lifetime in [decimal].

    Returns:
        :type:`float`: Total Net Present Cost of the converter in present value in [$].
    '''
    
    # Installation cost (CAPEX)
    installation_cost = self.cost_per_kw * (rated_power ** self.cost_scale)
    NPC = installation_cost
    # O&M costs (discounted)
    NPC += (installation_cost * maintenance_cost_rate) / CRF
    # Replacement costs (discounted)
    n_repl = np.ceil(project_lifetime_intervals / self.lifetime)
    NPC += np.sum(installation_cost * (n_repl[1:] - n_repl[:-1]) / ((1 + discount_rate) ** project_lifetime_intervals[1:]))
    return float(NPC)