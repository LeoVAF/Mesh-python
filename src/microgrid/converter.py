import numpy as np

class Converter():
  ''' Class to simulate the microgrid converter.
  
  Args:
    cost_per_kw (:type:`int | float`): Converter cost per kW of nominal capacity in [US$/kW].
    cost_scale (:type:`int | float`): Cost scale factor for converter, where a higher power results in a lower cost per kW in [decimal].
    efficiency (:type:`int | float`): Converter efficiency between 0 and 1.
    lifetime (:type:`int | float`): Converter lifetime in [year].
  
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
    ''' Converter cost per kW of nominal capacity in [US$/kW]. '''
    self.cost_scale: int | float
    ''' Scaling cost factor for converter, where a higher power results in a lower cost per kW in [decimal]. '''
    self.efficiency: int | float
    ''' Converter efficiency between 0 and 1. '''
    self.lifetime: int | float
    ''' Converter lifetime in [year]. '''
    self.operation_cost: float = 0.0
    ''' Total costs of the converter in the microgrid during the operation simulation in [US$]. '''

    self.cost_per_kw = cost_per_kw
    self.cost_scale = cost_scale
    self.efficiency = efficiency
    self.lifetime = lifetime

  def economic_analysis(self, rated_power: int | float , project_lifetime: int | float, maintenance_cost_rate: int | float, discount_rate: int | float) -> float:
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
        \text{NPV}_{om} = \sum^{T-1}_{t=0}\frac{\text{IC}_{conv} \cdot \tau_{om}}{(1 + d)^t}.

    :math:`T` is the project lifetime in [years], :math:`d` is the discount rate per year (assumed to be constant) in [decimal] and :math:`\tau_{om}` is the operation and maintenance cost rate in [decimal]. The replacement costs occur every :attr:`lifetime` years and are equal to the installation cost, discounted to present value according to the following equation:
    
    .. math::
        \text{NPV}_{repl} = \sum_{t \in T_{repl}}\frac{\text{IC}_{conv}}{(1 + d)^t},

    where :math:`T_{repl}` is the set of replacement years.

    Args:
        rated_power (:type:`int | float`): The power supported by the converter in [kW].
        project_lifetime (:type:`int | float`): Total project lifetime in [years].
        maintenance_cost_rate (:type:`int | float`): Operation and maintenance cost rate based on installation costs in [decimal].
        discount_rate (:type:`int | float`): Discount rate (per year) during the project lifetime in [decimal].

    Returns:
        :type:`float`: Total Net Present Cost of the converter in present value in [US$].
    '''
    
    years = np.arange(project_lifetime)
    # Installation cost (CAPEX)
    installation_cost = self.cost_per_kw * (rated_power ** self.cost_scale)
    NPC = installation_cost
    # O&M costs (discounted)
    OM_cost = (installation_cost * maintenance_cost_rate) / ((1 + discount_rate) ** years)
    NPC += np.sum(OM_cost)
    # Replacement costs (discounted)
    replacement_years = np.arange(self.lifetime, project_lifetime, self.lifetime)
    if len(replacement_years) > 0:
        NPV_repl = installation_cost / ((1 + discount_rate) ** replacement_years)
        NPC += np.sum(NPV_repl)
    return NPC