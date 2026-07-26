import numpy as np


class Inverter:
  ''' Class to simulate the microgrid DC/AC inverter.
  
  Args:
    reference_cost (:type:`float`): Inverter reference cost associated with a rated power of 1 kW in [$].
    cost_exponent (:type:`float`): Exponent cost factor for inverter, where a higher power results in a lower cost per kW in [decimal].
    efficiency (:type:`float`): Inverter efficiency between 0 and 1.
    lifetime (:type:`float`): Inverter lifetime in time intervals.
    resale_rate (:type:`float`): Resale rate of the inverter in [decimal].
    
  '''

  def __init__(self,
               reference_cost: float,
               cost_exponent: float = 0.95,
               efficiency: float = 0.95,
               lifetime: float = 10,
               resale_rate: float = 0.75):

    self.reference_cost: float
    ''' Inverter reference cost associated with a rated power of 1 kW in [$]. '''
    self.cost_exponent: float
    ''' Exponent cost factor for inverter, where a higher power results in a lower cost per kW in [decimal]. '''
    self.efficiency: float
    ''' Inverter efficiency between 0 and 1. '''
    self.lifetime: float
    ''' Inverter lifetime in time intervals. '''
    self.resale_rate: float
    ''' Resale rate of the inverter in [decimal]. '''

    self.reference_cost = reference_cost
    self.cost_exponent = cost_exponent
    self.efficiency = efficiency
    self.lifetime = lifetime
    self.resale_rate = resale_rate

  def economic_analysis(self,
                        rated_power: float,
                        project_lifetime_intervals: np.typing.NDArray[np.integer],
                        maintenance_cost_rate: float,
                        discount_rate: float,
                        CRF: float) -> float:
    r''' Performs the economic analysis of the inverter using the Net Present Cost (NPC) approach.

    The total NPC of the inverter is given by:

    .. math::
        NPC = \text{IC} + \text{NPV}_{om} + \text{NPV}_{repl} - \text{NPV}_{sv}.

    Where:
    
    - :math:`\text{IC}` is the installation cost;
    - :math:`\text{NPV}_{om}` is the Net Present Value of annual operation and maintenance costs;
    - :math:`\text{NPV}_{repl}` is the Net Present Value of replacement costs during the project lifetime;
    - :math:`\text{NPV}_{sv}` is the Net Present Value of the resale value (salvage value) of the inverter at the end of its useful life.

    The installation cost is calculated as:

    .. math::
        \text{IC} = C^{ref}_{inv} \cdot \left(\frac{P_{rated}}{1 \ \text{kW}}\right)^{\rho_{inv}}.

    :math:`C^{ref}_{inv}` is the inverter reference cost associated with a rated power of 1 kW [$], :math:`P_{rated}` is the rated power of the distributed energy resources and :math:`\rho_{inv}` is the inverter cost-scaling exponent. The operation and maintenance costs are calculated as:

    .. math::
        \text{NPV}_{om} = \sum^{T}_{t=1}\frac{\text{IC} \cdot \tau_{om}}{(1 + d)^t}.

    :math:`T` is the project lifetime in time intervals, :math:`d` is the discount rate per interval (assumed to be constant) in [decimal] and :math:`\tau_{om}` is the operation and maintenance cost rate in [decimal]. The replacement costs occur every :attr:`lifetime` intervals and are equal to the installation cost, discounted to present value according to the following equation:
    
    .. math::
      \text{NPV}_{repl} = \sum^{T}_{t=1}\frac{\left(\left\lfloor \frac{t}{T_{\text{repl}}} \right\rfloor - \left\lfloor \frac{t-1}{T_{\text{repl}}} \right\rfloor\right) \cdot \text{IC}}{(1 + d)^t},

    where :math:`T_{repl} = I^{\text{lifetime}}` is the time when the equipment must be replaced. The salvage value is calculated as:

    .. math::
      \text{NPV}_{sv} = \frac{\text{IC} \cdot \tau_{sv} \cdot T_{\text{remaining}}}{(1 + d)^T},

    where :math:`\tau_{sv}` is the resale rate of the inverter in [decimal] and :math:`T_{\text{remaining}}` is the remaining lifetime of the inverter in [decimal].

    Args:
        rated_power (:type:`float`): The power supported by the inverter in [kW].
        project_lifetime_intervals (:type:`np.typing.NDArray[np.integer]`): Intervals of project lifetime.
        maintenance_cost_rate (:type:`float`): Operation and maintenance cost rate based on installation costs in [decimal].
        discount_rate (:type:`float`): Discount rate (per interval) during the project lifetime in [decimal].
        CRF (:type:`float`): Capital Recovery Factor (CRF) during the project lifetime in [decimal].

    Returns:
        :type:`float`: Total Net Present Cost of the inverter in present value in [$].
    '''
    
    # Installation cost (CAPEX)
    installation_cost = self.reference_cost * (rated_power ** self.cost_exponent)
    NPC = installation_cost
    # O&M costs (discounted)
    NPC += (installation_cost * maintenance_cost_rate) / CRF
    # Replacement costs (discounted)
    n_repl = np.floor(project_lifetime_intervals / self.lifetime)
    NPC += np.sum(installation_cost * (n_repl[1:] - n_repl[:-1]) / ((1 + discount_rate) ** project_lifetime_intervals[1:]))
    # Resale | salvage value (discounted)
    project_lifetime = project_lifetime_intervals[-1]
    remaining_lifetime = 1 - (project_lifetime % self.lifetime) / self.lifetime
    NPC -= (installation_cost * self.resale_rate * remaining_lifetime) / ((1 + discount_rate) ** project_lifetime_intervals[-1])
    return float(NPC)