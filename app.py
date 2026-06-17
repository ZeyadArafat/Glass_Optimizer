import streamlit as st
import numpy as np
from scipy.optimize import minimize_scalar
import pandas as pd

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Egypt Glass Industry - Cullet Optimizer",
    layout="wide"
)

st.title("🎛️ Glass Furnace Cullet, Energy & Electricity Optimization Model")

st.markdown("""
This tool optimizes the cullet ratio by balancing:

- Raw material costs
- Cullet scarcity effects
- Natural gas consumption
- Electricity consumption

Assumption used:
**Every 10% increase in cullet reduces total furnace energy demand by 3%.**
""")

# --------------------------------------------------
# SIDEBAR INPUTS
# --------------------------------------------------

st.sidebar.header("🛠️ Model Input Parameters")

# ---------------------------
# Natural Gas
# ---------------------------
st.sidebar.subheader("Natural Gas Parameters")

p_ng = st.sidebar.number_input(
    "Natural Gas Cost ($/MMBtu)",
    min_value=1.0,
    max_value=20.0,
    value=6.75,
    step=0.25
)

sec_0 = st.sidebar.slider(
    "Baseline Furnace SEC at 0% Cullet (GJ/ton)",
    3.5,
    6.5,
    4.8,
    0.1
)

# ---------------------------
# Electricity
# ---------------------------
st.sidebar.subheader("Electricity Parameters")

p_elec = st.sidebar.number_input(
    "Electricity Price ($/kWh)",
    min_value=0.01,
    max_value=0.50,
    value=0.11,
    step=0.01
)

elec_base_kwh = st.sidebar.number_input(
    "Baseline Electricity Use (kWh/t Glass)",
    min_value=50.0,
    max_value=500.0,
    value=180.0,
    step=5.0
)

elec_saving_frac = st.sidebar.slider(
    "Share of Thermal Savings Affecting Electricity",
    min_value=0.0,
    max_value=1.0,
    value=0.40,
    step=0.05
)

# ---------------------------
# Material Costs
# ---------------------------
st.sidebar.subheader("Material Cost & Scarcity")

c_batch = st.sidebar.number_input(
    "Virgin Batch Cost ($/ton)",
    min_value=50.0,
    max_value=250.0,
    value=110.0,
    step=5.0
)

p_base_cullet = st.sidebar.number_input(
    "Base Cullet Cost ($/ton)",
    min_value=30.0,
    max_value=150.0,
    value=65.0,
    step=5.0
)

beta = st.sidebar.slider(
    "Cullet Scarcity Premium Coefficient (Beta)",
    min_value=20.0,
    max_value=200.0,
    value=95.0,
    step=5.0
)

# ---------------------------
# Technical Constraints
# ---------------------------
st.sidebar.subheader("Technical Constraints")

rc_max = st.sidebar.slider(
    "Maximum Technical Cullet Ratio",
    min_value=0.10,
    max_value=0.95,
    value=0.80,
    step=0.05
)

loi = (
    st.sidebar.slider(
        "Batch Loss on Ignition (%)",
        min_value=10.0,
        max_value=22.0,
        value=16.0,
        step=0.5
    )
    / 100
)

# --------------------------------------------------
# UNIT CONVERSIONS
# --------------------------------------------------

# Convert $/MMBtu to $/GJ
p_ng_gj = p_ng / 1.055056

# --------------------------------------------------
# MODEL FUNCTIONS
# --------------------------------------------------

def get_sec(Rc):
    """
    10% cullet saves 3% energy
    => 100% cullet saves 30%
    """
    energy_saving_fraction = 0.30 * Rc
    return sec_0 * (1 - energy_saving_fraction)


def get_cullet_price(Rc):
    """
    Scarcity premium model
    """
    return p_base_cullet + beta * (Rc ** 2)


def get_electricity_use(Rc):
    """
    Part of electricity demand benefits
    from the thermal efficiency improvement.
    """
    thermal_saving = 0.30 * Rc
    elec_saving = elec_saving_frac * thermal_saving

    return elec_base_kwh * (1 - elec_saving)


def get_material_cost(Rc):
    return ((1 - Rc) * c_batch) + (Rc * get_cullet_price(Rc))


def get_gas_cost(Rc):
    return get_sec(Rc) * p_ng_gj


def get_electricity_cost(Rc):
    return get_electricity_use(Rc) * p_elec


def objective_total_cost(Rc):
    return (
        get_material_cost(Rc)
        + get_gas_cost(Rc)
        + get_electricity_cost(Rc)
    )

# --------------------------------------------------
# OPTIMIZATION
# --------------------------------------------------

result = minimize_scalar(
    objective_total_cost,
    bounds=(0.0, rc_max),
    method="bounded"
)

# --------------------------------------------------
# RESULTS
# --------------------------------------------------

if result.success:

    opt_Rc = result.x

    min_cost = result.fun

    baseline_cost = objective_total_cost(0.0)

    saved_cost = baseline_cost - min_cost

    opt_sec = get_sec(opt_Rc)

    opt_cullet_price = get_cullet_price(opt_Rc)

    opt_elec_use = get_electricity_use(opt_Rc)

    eta_batch = 1.0 - loi

    batch_req = (1.0 - opt_Rc) / eta_batch

    # ---------------------------
    # KPI DISPLAY
    # ---------------------------

    st.subheader("📊 Optimal Furnace Operating Point")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Optimal Cullet Ratio",
            f"{opt_Rc * 100:.1f}%"
        )

    with col2:
        st.metric(
            "Energy Intensity",
            f"{opt_sec:.2f} GJ/t"
        )

    with col3:
        st.metric(
            "Electricity Use",
            f"{opt_elec_use:.1f} kWh/t"
        )

    with col4:
        st.metric(
            "Cullet Market Price",
            f"${opt_cullet_price:.2f}/t"
        )

    with col5:
        st.metric(
            "Minimum OPEX",
            f"${min_cost:.2f}/t",
            delta=f"${saved_cost:.2f} Saved"
        )

    st.markdown("---")

    # --------------------------------------------------
    # CURVE ANALYSIS
    # --------------------------------------------------

    st.subheader("📈 Cost-Benefit Analysis")

    rc_range = np.linspace(0, rc_max, 100)

    energy_costs = [get_gas_cost(r) for r in rc_range]

    electricity_costs = [
        get_electricity_cost(r)
        for r in rc_range
    ]

    material_costs = [
        get_material_cost(r)
        for r in rc_range
    ]

    total_costs = [
        get_gas_cost(r)
        + get_electricity_cost(r)
        + get_material_cost(r)
        for r in rc_range
    ]

    chart_df = pd.DataFrame({
        "Cullet Ratio (%)": rc_range * 100,
        "Natural Gas Cost ($/t)": energy_costs,
        "Electricity Cost ($/t)": electricity_costs,
        "Material Cost ($/t)": material_costs,
        "Total Production Cost ($/t)": total_costs
    })

    st.line_chart(
        chart_df,
        x="Cullet Ratio (%)",
        y=[
            "Natural Gas Cost ($/t)",
            "Electricity Cost ($/t)",
            "Material Cost ($/t)",
            "Total Production Cost ($/t)"
        ]
    )

    # --------------------------------------------------
    # SAVINGS BREAKDOWN
    # --------------------------------------------------

    st.subheader("💰 Savings Breakdown")

    baseline_gas = get_gas_cost(0.0)
    baseline_elec = get_electricity_cost(0.0)

    optimized_gas = get_gas_cost(opt_Rc)
    optimized_elec = get_electricity_cost(opt_Rc)

    gas_savings = baseline_gas - optimized_gas
    elec_savings = baseline_elec - optimized_elec

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Gas Cost Savings",
            f"${gas_savings:.2f}/t"
        )

    with c2:
        st.metric(
            "Electricity Cost Savings",
            f"${elec_savings:.2f}/t"
        )

    with c3:
        st.metric(
            "Total Savings vs 0% Cullet",
            f"${saved_cost:.2f}/t"
        )

    st.markdown("---")

    # --------------------------------------------------
    # ESG / SUSTAINABILITY
    # --------------------------------------------------

    st.subheader("🌱 Sustainability & Operations")

    col_a, col_b = st.columns(2)

    with col_a:

        virgin_material_reduction = opt_Rc * 100

        st.markdown(f"""
### Batch Plant Impact

- Virgin raw material requirement:
  **{batch_req:.3f} tons per ton of glass**

- Virgin material displacement:
  **{virgin_material_reduction:.1f}%**

- Lower material handling and logistics burden.
""")

    with col_b:

        energy_drop = sec_0 - opt_sec

        st.markdown(f"""
### Decarbonization Metrics

- Thermal energy reduction:
  **{energy_drop:.2f} GJ/t Glass**

- Electricity reduction:
  **{elec_base_kwh - opt_elec_use:.1f} kWh/t Glass**

- Cullet utilization:
  **{opt_Rc * 100:.1f}%**

- Estimated process-emission reduction:
  **{opt_Rc * 100:.1f}%**
""")

else:
    st.error(
        "Optimization failed. Please adjust the input parameters."
    )