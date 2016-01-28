# Enron Data Notes

[About Deferred Income plans at Enron](https://books.google.com/books?id=yhSA2u91BFgC&pg=PA619&lpg=PA619&dq=enron+deferred+income&source=bl&ots=Nb_6vHc9qs&sig=G6yg8j_rJUIVFaWP7W0AMsrgdIg&hl=en&sa=X&ved=0ahUKEwiUjvKs5MrKAhURxWMKHWS1DLMQ6AEIIzAB#v=onepage&q=enron%20deferred%20income&f=false)

[About Deferred Revenue in General](http://www.investopedia.com/terms/d/deferredrevenue.asp)

The deferred income is money that the person would have gotten that was voluntarily surrendered from their salary. Deferral payments represents the money an employee withdrew from that deferred income account or that they got on termination of employment.

Robert A Belfer has an incorrect input. His "deferred income" data is for some reason in the "deferral payments" column, according to the included FindLaw PDF.

If you consider that "deferred income" is money that the employee in question WOULD HAVE earned

Note deferral payments suffer a 10% forfeiture for early withdrawal. We should add that back in

Sanjay Bhatnagar's data is also suspect. `data_dict` notes that he has $15,456,290 in restricted_stock_deferred but this is actually the amount of his `exercised_stock_options`. 

Clearly, I need to go through this data set name by name and check things since the tabled data and the dictionary data aren't matching up. Unfortunately there's so many naems, so I'll instead just have to keep an eye out for anomalies. Here's the list of employees with problematic data:

- BELFER ROBERT (restricted stock, restricted stock deferred, deferral payments,  deferred income, exercised stock options)
- BHATNAGAR SANJAY (total payments, exercised stock options, restricted stock, restricted stock deferred, total stock value, other, expenses, director_fees, )
- MARTIN AMANDA K (email_address maybe)
- MULLER MARK S (email_address maybe)

Columns I don't need either due to lack of data or just general uselessness:

- director_fees
- loan_advances
- restricted_stock_deferred
- restricted_stock

If I'm using both email data and financial data, I should rescale the values. I should also rescale the stocks/cash ratio