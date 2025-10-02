import pandas as pd, time

#Check the book database for the container number of the book that the user has requested.
def checkBookData(userRequest):
    
    bdf = pd.read_csv('BookDatabase.csv')

    # Filter the DataFrame
    result = bdf[bdf["Book"].str.contains(userRequest)]

    containerNum = result["Container Number"]
    
    if containerNum.empty:
        return 0
    
    containerNum = int(containerNum.iloc[0])
    return containerNum
    
#Check database for coordinates to navigate to. Move motors to align belts with location.
def navigateGantry(location):

    #print("Request container",location)
    
    ldf = pd.read_csv('LocationDatabase.csv')
    
    # Filter the DataFrame
    result = ldf[ldf["Location"].str.contains(str(location))]
    
    xCoord = result.iloc[0]["X"]
    yCoord = result.iloc[0]["Y"]
    
    print("Moving to",location,"at X=",xCoord,"and Y=",yCoord)
    
    time.sleep(1.5) #Sleep command to represent gantry moving

def retrieveContainer():
    print("End effector retrieving container")
    
    time.sleep(1) #Sleep command to represent end effector moving
    
def depositContainer():
    print("End effector depositing container")
    
    time.sleep(1) #Sleep command to represent end effector moving
    
#def updateDatabase():

#Prompt user for book to retrieve
print("Please enter book to retrieve: ")
userRequest = input()

#Get container number for book
containerNum = checkBookData(userRequest)

if containerNum == 0:
    print("Don't got that sorry")
else:
    #Navigate to container and retrieve it
    print(userRequest,"is in container",containerNum)
    navigateGantry(containerNum)
    retrieveContainer()
    
    #Navigate to dropoff and deposit book
    navigateGantry("Dropoff")
    depositContainer()
