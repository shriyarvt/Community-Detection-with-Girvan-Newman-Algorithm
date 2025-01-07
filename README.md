This project implements the Girvan-Newman algorithm for community detection within a network graph using Spark RDD and standard 
Python libraries. The goal is to detect and output communities that achieve the highest modularity, dividing the graph into meaningful 
subgraphs. The algorithm iteratively removes edges with the highest betweenness and recomputes the betweenness after each removal. 
This continues until the graph is divided into communities that maximize the modularity of the network. The output consists of both the 
betweenness file and the detected communities, ensuring that the graph is divided in a way that reflects the highest possible modularity. 
