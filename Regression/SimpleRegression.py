#-------------------------------------------------------------------------------
# Name:        Regression Week 1 Programming Assignment
# Purpose:
#
# Author:      Nonlining
#
# Created:     15/08/2016
# Copyright:   (c) Nonlining 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import graphlab

train_data = 0
test_data = 0
sales = 0

def inverse_regression_predictions(output, intercept, slope):
    # solve output = intercept + slope*input_feature for input_feature. Use this equation to compute the inverse predictions:
    estimated_feature = float(output - intercept)/float(slope)

    return estimated_feature


def get_regression_predictions(input_feature, intercept, slope):
    # calculate the predicted values:

    predicted_values = intercept + slope*input_feature

    return predicted_values

def get_residual_sum_of_squares(input_feature, output, intercept,slope):
    #[your code here]
    gap = (output - (intercept + slope*input_feature))
    RSS = (gap*gap).sum()
    return(RSS)


def simple_linear_regression(input_feature, output):
    #[your code here]
    # compute the sum of input_feature and output
    sum_of_input = input_feature.sum()
    sum_of_output = output.sum()
    # compute the product of the output and the input_feature and its sum
    sum_product_of_in_out = (input_feature*output).sum()

    # compute the squared value of the input_feature and its sum
    sum_squared_input = (input_feature * input_feature).sum()


    # use the formula for the slope
    slope = float(sum_product_of_in_out - float(sum_of_input*sum_of_output)/float(input_feature.size()) )/float(sum_squared_input - float(sum_of_input*sum_of_input)/float(input_feature.size()) )

    # use the formula for the intercept

    intercept = float(sum_of_output)/float(output.size()) - slope*float(sum_of_input)/float(output.size())

    return(intercept, slope)


def loaddata():
    global sales, train_data, test_data
    sales = graphlab.SFrame('kc_house_data.gl/')
    train_data,test_data = sales.random_split(.8,seed=0)

def main():

    prices = sales['price']
    print len(prices)
    # recall that the arithmetic average (the mean) is the sum of the prices divided by the total number of houses:
    sum_prices = prices.sum()
    num_houses = prices.size() # when prices is an SArray .size() returns its length
    avg_price_1 = sum_prices/num_houses
    avg_price_2 = prices.mean() # if you just want the average, the .mean() function
    print "average price via method 1: " + str(avg_price_1)
    print "average price via method 2: " + str(avg_price_2)


def week1demo():
    test_feature = graphlab.SArray(range(5))
    test_output = graphlab.SArray(1 + 1*test_feature)
    print "====testing data set===="
    print test_feature
    print test_output
    (test_intercept, test_slope) =  simple_linear_regression(test_feature, test_output)
    print "Intercept: " + str(test_intercept)
    print "Slope: " + str(test_slope)
    print "====Simple Linear Regression===="
    sqft_intercept, sqft_slope = simple_linear_regression(train_data['sqft_living'], train_data['price'])

    print "Intercept: " + str(sqft_intercept)
    print "Slope: " + str(sqft_slope)
    print "Assignment Q1 answer : "
    my_house_sqft = 2650
    estimated_price = get_regression_predictions(my_house_sqft, sqft_intercept, sqft_slope)
    print "The estimated price for a house with %d squarefeet is $%.2f" % (my_house_sqft, estimated_price)

    print "Assignment Q2 answer : "

    rss_prices_on_sqft = get_residual_sum_of_squares(train_data['sqft_living'], train_data['price'], sqft_intercept, sqft_slope)
    print 'The RSS of predicting Prices based on Square Feet is : ' + str(rss_prices_on_sqft)

    print "Assignment Q3 answer : "
    my_house_price = 800000
    estimated_squarefeet = inverse_regression_predictions(my_house_price, sqft_intercept, sqft_slope)
    print "The estimated squarefeet for a house worth $%.2f is %d" % (my_house_price, estimated_squarefeet)

    print "====New model feature - bedrooms===="
    br_intercept, br_slope = simple_linear_regression(train_data['bedrooms'], train_data['price'])
    rss_prices_on_bedroom = get_residual_sum_of_squares(train_data['bedrooms'], train_data['price'], br_intercept, br_slope)
    print 'The RSS of predicting Prices based on Bedrooms is : ' + str(rss_prices_on_bedroom)


if __name__ == '__main__':

    loaddata()
    main()
    week1demo()
