import aips.lab07.solution.nQueen as queen
import aips.local.search as search

def run():
    
    state=queen.NqState(4,None)     #create a 4-queen state

    #test str() method
    print("---str() method returns:---\n{}".format(state))

    #test neighbours() method
    neighbours=state.neighbours()
    print("---Number of neighbours returned by neighbours() method: {}---".format(len(neighbours)))

    print("---Neighours found:---");
    
    for n in neighbours:
        print(n)
        
if __name__=="__main__":
    run()
    