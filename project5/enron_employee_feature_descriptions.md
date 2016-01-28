#Enron Employee Features Explanation

This file will list and describe all of the features in the Enron Employee features list

## Features

### poi
A description of whether or not the person in question is a known person of interest | boolean

### salary
The total annual salary ($USD) of the employee | float

### to_messages
Number of messages to the employee | int

### deferral_payments
Voluntary executive deferrals of salary, annual cash incentives, and long-term cash incentives as well as cash fees deferred by non-employee directors under a deferred compensation arrangement. May also reflect deferrals under a stock option or phantom stock unit in lieu of cash arrangement | float

### total_payments
The financial total of all payments | float

### exercised_stock_options
Exercised stock options which equal the market value in excess of the exercise price on the date the options were exercised through cashless, stock swap, or cash exercises | float

### bonus
Annual cash incentives, as well as other retention payments | float

### restricted_stock
The gross fair market value of shares and accrued dividends (and/or phantom units and dividend equivalents) on the date of release | float

### shared_receipt_with_poi
Number of messages where this person was on the same email list as a known POI | float

### restricted_stock_deferred
Value of restricted stock voluntarily deferred | float

### total_stock_value
Total of exercised, restricted, and restricted-deferred stock | float

### expenses
Reimbursements of business expenses. May include fees paid for consulting services | float

### loan_advances
Total amount of loan advances, excluding repayments, provided by the Debtor in return for a promise of repayment. In certain instances, the terms of the promissory notes allow for the option to repay with stock of the company | float

### from_messages
Total number of messages from this employee to others | int

### other
Other amounts of money such as payments for severance, consulting services, relocation costs, tax advances and allowances for employees on international assignment (e.g. housing allowances, cost of living allowances, payments under Enron's Tax Equalization Program, etc). May also include payments provided with respect to employment agreements, as well as imputed income amounts for such things as use of corporate aircraft | float

### from_this_person_to_poi
Number of messages from this employee to a person of interest | int

### poi
Is this a known person of interest or not? | boolean

### director_fees
Cash payments and/or value of stock grants made in lieu of cash payments to non-employee directors | float

### deferred_income
Voluntary executive deferrals of salary, annual cash incentives, and long-term cash incentive as well as cash fees deferred by non-employee directors under a deferred compensation arrangement. May also reflect deferrals under a stock option or phantom stock unit in lieu of cash arrangement | float

### long_term_incentive
Like the above but for long term incentives tying executive compensation to long-term success as measured against key performance drivers and business objectives over a multi-year period (3-5 years) | float

### email_address
The email address of the employee | str

### from_poi_to_this_person
The total number of emails from a person of interest to this employee | int
