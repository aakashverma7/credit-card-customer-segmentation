# Data Notes

Data not included in this repository.

Place the source file in this folder as a CSV or Excel workbook. Common names:

- `credit_card_customers.xlsx`
- `credit_card_customers.csv`

Expected columns include:

- `CUST_ID`
- `BALANCE`
- `PURCHASES`
- `ONEOFF_PURCHASES`
- `INSTALLMENTS_PURCHASES`
- `CASH_ADVANCE`
- `CREDIT_LIMIT`
- `PAYMENTS`
- `MINIMUM_PAYMENTS`
- `PRC_FULL_PAYMENT`
- `TENURE`

The script will use every numeric feature that is present, but it expects at least:

- `CREDIT_LIMIT`
- `PAYMENTS`
- `MINIMUM_PAYMENTS`
