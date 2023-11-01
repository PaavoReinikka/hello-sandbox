#Initial solution by Veli Kangas MDM2021
# Input graph modifies by Paavo, for MDM2023 task 4.3

# Node labels for all six graphs, ordered left-to-right, top-to-bottom. Nodes are identified by their index here\n",
V = [['C', 'C', 'C', 'N', 'C', 'C', 'C', 'O','O'],#G1
     ['C', 'C', 'C', 'N', 'C', 'C', 'C', 'N', 'C', 'C', 'C','C'],#G2
     ['C', 'C', 'C', 'C', 'C', 'C', 'O', 'C', 'C', 'N', 'C', 'C', 'N', 'C', 'C'],#G3
     ['C', 'C', 'N', 'C', 'C', 'C', 'O', 'O']]
# Edges as tuples of node indices
E = [[(0,1),(0,5),(1,2),(1,6),(2,3),(3,4),(4,5),(6,7),(6,8)],
     [(0,1),(0,5),(1,2),(1,6),(2,3),(3,4),(4,5),(6,7),(6,10),(7,8),(7,11),(8,9),(9,10)],
     [(0,1),(0,5),(0,6),(1,2),(1,7),(2,3),(2,9),(3,4),(4,5),(7,8),(7,10),(8,9),(10,11),(11,12),(12,13),(12,14)],
     [(0,1),(0,4),(1,2),(1,5),(2,3),(3,4),(5,6),(5,7)]]

def MCG(V1, E1, V2, E2):
    n1 = len(V1) # Size of G1
    n2 = len(V2) # Size of G2
    
    C = list()   # List of all label-matching node pairs
    for i in range(n1):
        for j in range(n2):
            if (V1[i] == V2[j]):
                C.append((i,j)) # Add all pairs with matching labels
                
    VM1 = list()      # Currently matched nodes from G1
    VM2 = list()      # Corresponding nodes from G2
    VM1_best = list() # Matched nodes from G1 in current best solution
    VM2_best = list() # Corresponding nodes from G2
    
    return MCG_rec(V1, E1, V2, E2, VM1, VM2, VM1_best, VM2_best, C)

def MCG_rec(V1, E1, V2, E2, VM1, VM2, VM1_best, VM2_best, C):
    # Go through all label-matching node pairs
    for c in C:
        # Skip if either node has already been matched (= included in the subgraph)
        if (c[0] in VM1 or c[1] in VM2):
            continue
            
        conn1 = list() # All matched nodes that the newly added node is connected to in G1
        # For each edge in G1, if one end point is the new node and
        # the other is matched, add the matched one to conn1
        for e in E1:
            if e[0] == c[0] and e[1] in VM1:
                conn1.append(e[1])
            elif e[1] == c[0] and e[0] in VM1:
                conn1.append(e[0])
                
        # Same for G2
        conn2 = list()
        for e in E2:
            if e[0] == c[1] and e[1] in VM2:
                conn2.append(e[1])
            elif e[1] == c[1] and e[0] in VM2:
                conn2.append(e[0])
                
        # Match is not valid if new node is connected to different numbers of matched nodes
        # in original graphs, or if either is disconnected from the current subgraph
        if (len(conn1) != len(conn2) or len(VM1) > 0 and (len(conn1) == 0 or len(conn2) == 0)):
            continue
            
        # For each node in conn1, check that the corresponding node is in conn2
        # (new node is connected to the same node in G1 and G2).
        # As lengths are equal and no duplicates, it's enough to do this one way
        matching_edges = True
        for v in conn1:
            i = VM1.index(v) # index of the pairing containing node v from G1
            if VM2[i] in conn2:
                continue
            else:
                matching_edges = False
                break
        
        # If new node was not connected to same matched nodes in both G1 and G2, match is not valid
        if matching_edges == False:
            continue

        # If all is good, add the new matched pair and recurse deeper
        VM1.append(c[0])
        VM2.append(c[1])
        VM1_best, VM2_best = MCG_rec(V1, E1, V2, E2, VM1, VM2, VM1_best, VM2_best, C)
        
        # Remove the pair that was added as this branch has been searched
        n = len(VM1)
        VM1.pop(n-1)
        VM2.pop(n-1)
           
    # If the match is larger than current maximum, update        
    if len(VM1) > len(VM1_best):
        VM1_best = VM1.copy()
        VM2_best = VM2.copy()
        
    return VM1_best, VM2_best
          
    
N = len(V) # number of graphs
for i in range(N):
    for j in range(i+1,N):
        print("-----------------------------------------")
        print("Finding MCG for graphs", i+1, "and", j+1)
        VM1_best, VM2_best = MCG(V[i], E[i], V[j], E[j])
        print("Subgraph nodes in G1:", VM1_best)
        print("Matching nodes in G2:", VM2_best)
        print("MCG size:", len(VM1_best))
        print("UDist(G1,G2) = ", round(1 - len(VM1_best) / (len(V[i]) + len(V[j]) - len(VM1_best)),3))
        print("MDist(G1,G2) = ", round(1 - len(VM1_best) / max(len(V[i]), len(V[j])),3))
