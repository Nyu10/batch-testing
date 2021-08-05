import random
import statistics

population = 10000
batch_size = 10
infection_rate = 0.01
type2_error = 0.15
type1_error = 0.01

one_test_list = []
two_test_list = []
#
#simulates PCR Test by returning if the batch is negative or positive
def is_batch_infected(batch_instance):
    testing_error = random.random()
    #false is negative, True is positive
    batch_boolean = False
    for person in batch_instance:
        if person == "infected":
            batch_boolean=True
    #if the batch is infected, check if there is Type 2 Error (false negative)
    #if batch is not infected, check if there is Type 1 Error (false positive)
    if batch_boolean:
        return True if testing_error >= type2_error else False
    else:
        return False if testing_error >= type1_error else True

#100 simulations
for i in range(100):
    # setup population array with "infected" or "not infected"
    population_list = []
    for person in range(population):
        is_infected = random.random()
        if is_infected <= infection_rate:
            population_list.append("infected")
        else:
            population_list.append("not_infected")

    num_of_one_test = 0
    num_of_two_test = 0
    for batch_num in range(0, population, batch_size):
        # batch_instance is 10 individuals
        batch_instance = population_list[batch_num:batch_num+batch_size]
        if is_batch_infected(batch_instance):
            num_of_two_test += 10
        else:
            num_of_one_test += 10
    one_test_list.append(num_of_one_test)
    two_test_list.append(num_of_two_test)

#average of 100 simulations of number of individuals who took one test
print("number of individuals who took one test")
print (sum(one_test_list)/len(one_test_list))

print("Standard Deviation of number of individuals who took one test")
print(statistics.pstdev(one_test_list))
#average of 100 simulations of number of individuals who took two tests
print("number of individuals who took two tests")
print(sum(two_test_list)/len(two_test_list))
print("Standard Deviation of number of individuals who took two tests")
print(statistics.pstdev(two_test_list))

