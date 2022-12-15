# Questions

This is a very simple question answering program. It uses term frequency-inverse document frequency (tf-idf) to search through a limited number of documents and passages and retrieve information that is the most pertinent to a query from the user. Each document is tokenized to determine the inverse document frequency (idf) for each word in the document. Based on the words in the query, the top documents are chosen based on their tf-idf score. The score is calculated by summing (frequency*idf score) for each word in the query that is found in the document. The top sentence from that document is determined using the sum of each words idf score from the query. Term frequency is not a factor when selecting for the most pertinent sentence. 

An introduction to AI with python project by Harvard University Online
