import streamlit as st

st.page_link("app.py", label="Back Home", icon="↩")

st.markdown("---")

st.title("How to Prepare the Input File")

st.markdown("""
This application requires a single Excel file containing three sheets:
**Client data**, **Courier data**, and **Weights**.
Each sheet must follow a strict structure so the optimizer can read the data correctly.

---

## 1. Sheet: Client data

This sheet describes how customers behave under different client price diffs.

### Required columns
| Column | Description |
|--------|-------------|
| **Distance bin** | Name or ID of the distance segment. Must match all other sheets. |
| **Client price diff** | Client-side price adjustment (for example -0.1, 0.0, 0.2). |
| **Demand conversion rate** | Probability that the customer places an order. Value between 0 and 1. |
| **Client payment** | Estimated client payment for this diff. Numeric. |

### Rules
- Every distance bin must contain a full grid of allowed client diffs.
- No missing values and no empty rows inside the data.
- Distance bin names must exactly match those in **Courier data** and **Weights**.
- All numerical columns must contain numbers only.

---

## 2. Sheet: Courier data

This sheet describes courier behavior and payouts for different courier price diffs.

### Required columns
| Column | Description |
|--------|-------------|
| **Distance bin** | Must match Client data. |
| **Courier price diff** | Courier payout adjustment. |
| **Completion rate** | Orders completion rate. Value between 0 and 1. |
| **Courier payment** | Estimated payout for this diff. Numeric. |

### Rules
- Every distance bin must have all the necessary courier diffs.
- Completion rate must be between 0 and 1.
- No partially filled rows.
- Distance bin names must match the other sheets exactly.

---

## 3. Sheet: Weights

This sheet defines the relative importance of each distance bin (for example, demand share).

### Required columns
| Column | Description |
|--------|-------------|
| **Distance bin** | Same naming as in the other sheets. |
| **Weight** | Numeric weight for this bin (for example, share of intents or revenue). |

### Rules
- The sheet must list every distance bin used in Client data and Courier data.
- All weights must be numeric.
- The sum of weights is typically close to 1, but this is not strictly required.

---

## General requirements for the Excel file

- File format must be **.xlsx**.
- Sheet names must be exactly:
  - `Client data`
  - `Courier data`
  - `Weights`
- Column names must match exactly (case sensitive).
- No empty header rows and no merged cells.
- Distance bin names must be identical across all sheets.
- Numerical values must be valid numbers.
- Conversion and completion rates must be between 0 and 1.

---

## Minimal file structure

Client data:
Distance bin | Client price diff | Demand conversion rate | Client payment

Courier data:
Distance bin | Courier price diff | Completion rate | Courier payment

Weights:
Distance bin | Weight


If your file follows this format, it will load correctly into the optimizer.

---
""")

st.page_link("app.py", label="Back Home", icon="↩")