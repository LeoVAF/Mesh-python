import numpy as np

class Converter():
  ''' Class to simulate the microgrid converter.
  
  Args:
    cost_per_kw (:type:`int | float`): Converter cost per kW of nominal capacity in [US$/kW].
    om_cost_rate (:type:`int | float`): Operation and maintenance cost rate for converter based on installation costs in [decimal].
    scaling_cost (:type:`int | float`): Scaling cost factor for converter, where a higher power results in a lower cost per kW in [decimal].
    efficiency (:type:`int | float`): Converter efficiency between 0 and 1.
    lifetime (:type:`int | float`): Converter lifetime in [year].
  
  Raises:
    TypeError: If the input is not the expected type.
    ValueError: If the input is not the allowed value.
  '''

  def __init__(self,
               cost_per_kw: int | float,
               om_cost_rate: int | float = 0.02,
               scalign_cost: int | float = 0.95,
               efficiency: int | float = 0.95,
               lifetime = 10):

    self.cost_per_kw: int | float
    ''' Converter cost per kW of nominal capacity in [US$/kW]. '''
    self.om_cost_rate: int | float
    ''' Operation and maintenance cost rate for converter based on installation costs in [decimal]. '''
    self.scaling_cost: int | float
    ''' Scaling cost factor for converter, where a higher power results in a lower cost per kW in [decimal]. '''
    self.efficiency: int | float
    ''' Converter efficiency between 0 and 1. '''
    self.lifetime: int | float
    ''' Converter lifetime in [year]. '''
    self.operation_cost: float = 0.0
    ''' Total costs of the converter in the microgrid during the operation simulation in [US$]. '''

    self.cost_per_kw = cost_per_kw
    self.om_cost_rate = om_cost_rate
    self.scaling_cost = scalign_cost
    self.efficiency = efficiency
    self.lifetime = lifetime

  def economic_analysis(self, rated_power: int | float , project_lifetime: int, discount_rate: int | float) -> float:
    r''' Performs the economic analysis of the converter using the Net Present Cost (NPC) approach.

    The total NPC of the converter is given by:

    .. math::
        NPC_{inv} = IC_{inv} + NPV_{OM} + NPV_{repl}.

    Where:
      - :math:`IC_{inv}` is the installation cost;
      - :math:`NPV_{OM}` is the Net Present Value of annual operation and maintenance costs;
      - :math:`NPV_{repl}` is the Net Present Value of replacement costs during the project lifetime.

    The installation cost is calculated as:

    .. math::
        IC_{inv} = C_{kW} \cdot P_{rated}^{C_s}.

    :math:`C_{kW}` is the cost per kW of nominal capacity for the converter, :math:`P_{rated}` is the rated power of the distributed energy resources and :math:`C_s` is the converter economies of scale. The operation and maintenance costs are assumed to be constant each year as a percentage of the installation cost:

    .. math::
        OM_{annual} = IC_{inv} \cdot \tau_{OM}.

    :math:`\tau_{OM}` is the operation and maintenance cost rate in [decimal]. The replacement costs occur every :attr:`lifetime` years and are equal to the initial installation cost,
    discounted to present value.

    Args:
        rated_power (:type:`int | float`): The power supported by the converter in [kW].
        project_lifetime (:type:`int`): Total project lifetime in [years].
        discount_rate (:type:`int | float`): Discount rate (per year) during the project lifetime in [decimal].

    Returns:
        :type:`float`: Total Net Present Cost of the converter in present value in [US$].
    '''
    
    years = np.arange(project_lifetime)
    # Installation cost (CAPEX)
    installation_cost = self.cost_per_kw * (rated_power ** self.scaling_cost)
    NPC = installation_cost
    # O&M costs (discounted)
    OM_cost = (self.om_cost_rate * installation_cost) / ((1 + discount_rate) ** years)
    NPC += np.sum(OM_cost)
    # Replacement costs (discounted)
    replacement_years = np.arange(self.lifetime, project_lifetime, self.lifetime)
    if len(replacement_years) > 0:
        NPV_repl = installation_cost / ((1 + discount_rate) ** replacement_years)
        NPC += np.sum(NPV_repl)
    return NPC