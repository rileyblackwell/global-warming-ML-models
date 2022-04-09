# -*- coding: utf-8 -*-
# Problem Set 5: Experimental Analysis
# Name: 
# Collaborators (discussion):
# Time:

import pylab
import re

# cities in our weather data
CITIES = [
    'BOSTON',
    'SEATTLE',
    'SAN DIEGO',
    'PHILADELPHIA',
    'PHOENIX',
    'LAS VEGAS',
    'CHARLOTTE',
    'DALLAS',
    'BALTIMORE',
    'SAN JUAN',
    'LOS ANGELES',
    'MIAMI',
    'NEW ORLEANS',
    'ALBUQUERQUE',
    'PORTLAND',
    'SAN FRANCISCO',
    'TAMPA',
    'NEW YORK',
    'DETROIT',
    'ST LOUIS',
    'CHICAGO'
]

TRAINING_INTERVAL = range(1961, 2010)
TESTING_INTERVAL = range(2010, 2016)

"""
Begin helper code
"""
class Climate(object):
    """
    The collection of temperature records loaded from given csv file
    """
    def __init__(self, filename):
        """
        Initialize a Climate instance, which stores the temperature records
        loaded from a given csv file specified by filename.

        Args:
            filename: name of the csv file (str)
        """
        self.rawdata = {}

        f = open(filename, 'r')
        header = f.readline().strip().split(',')
        for line in f:
            items = line.strip().split(',')

            date = re.match('(\d\d\d\d)(\d\d)(\d\d)', items[header.index('DATE')])
            year = int(date.group(1))
            month = int(date.group(2))
            day = int(date.group(3))

            city = items[header.index('CITY')]
            temperature = float(items[header.index('TEMP')])
            if city not in self.rawdata:
                self.rawdata[city] = {}
            if year not in self.rawdata[city]:
                self.rawdata[city][year] = {}
            if month not in self.rawdata[city][year]:
                self.rawdata[city][year][month] = {}
            self.rawdata[city][year][month][day] = temperature
            
        f.close()

    def get_yearly_temp(self, city, year):
        """
        Get the daily temperatures for the given year and city.

        Args:
            city: city name (str)
            year: the year to get the data for (int)

        Returns:
            a 1-d pylab array of daily temperatures for the specified year and
            city
        """
        temperatures = []
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        for month in range(1, 13):
            for day in range(1, 32):
                if day in self.rawdata[city][year][month]:
                    temperatures.append(self.rawdata[city][year][month][day])
        return pylab.array(temperatures)

    def get_daily_temp(self, city, month, day, year):
        """
        Get the daily temperature for the given city and time (year + date).

        Args:
            city: city name (str)
            month: the month to get the data for (int, where January = 1,
                December = 12)
            day: the day to get the data for (int, where 1st day of month = 1)
            year: the year to get the data for (int)

        Returns:
            a float of the daily temperature for the specified time (year +
            date) and city
        """
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        assert month in self.rawdata[city][year], "provided month is not available"
        assert day in self.rawdata[city][year][month], "provided day is not available"
        return self.rawdata[city][year][month][day]

def se_over_slope(x, y, estimated, model):
    """
    For a linear regression model, calculate the ratio of the standard error of
    this fitted curve's slope to the slope. The larger the absolute value of
    this ratio is, the more likely we have the upward/downward trend in this
    fitted curve by chance.
    
    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d pylab array of values estimated by a linear
            regression model
        model: a pylab array storing the coefficients of a linear regression
            model

    Returns:
        a float for the ratio of standard error of slope to slope
    """
    assert len(y) == len(estimated)
    assert len(x) == len(estimated)
    EE = ((estimated - y)**2).sum()
    var_x = ((x - x.mean())**2).sum()
    SE = pylab.sqrt(EE/(len(x)-2)/var_x)
    return SE/model[0]

"""
End helper code
"""

def generate_models(x, y, degs):
    """
    Generate regression models by fitting a polynomial for each degree in degs
    to points (x, y).

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        degs: a list of degrees of the fitting polynomial

    Returns:
        a list of pylab arrays, where each array is a 1-d array of coefficients
        that minimizes the squared error of the fitting polynomial
    """
    # Creates a model of the given degree.
    models = []
    for degree in degs:
        model = pylab.polyfit(x, y, degree)
        models.append(model)
    return models   

def r_squared(y, estimated):
    """
    Calculate the R-squared error term.
    
    Args:
        y: 1-d pylab array with length N, representing the y-coordinates of the
            N sample points
        estimated: an 1-d pylab array of values estimated by the regression
            model

    Returns:
        a float for the R-squared error term
    """
    # Calculates the numerator: sum of the squared differences for the estimated and the actual values.
    estimate_error = ((y - estimated)**2).sum() 
    # Calculates the denominator: sum of the squared differences for the actual values and the mean.
    mean = sum(y) / len(y)
    mean_error = ((y - mean)**2).sum()
    # Calculates the R squared value: 1 minus the estimated difference divided by the mean difference.
    return 1 - (estimate_error / mean_error)
     
def evaluate_models_on_training(x, y, models):
    """
    For each regression model, compute the R-squared value for this model with the
    standard error over slope of a linear regression line (only if the model is
    linear), and plot the data along with the best fit curve.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (aka model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        R-square of your model evaluated on the given data points,
        and SE/slope (if degree of this model is 1 -- see se_over_slope). 

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a pylab array storing the coefficients of
            a polynomial.

    Returns:
        None
    """
    for model in models:
        # Calculates estimated y values(temps in degrees Celsius) from the models polynomial coefficients.
        est_y = pylab.polyval(model, x)
        r2 = r_squared(y, est_y)
        # Creates a figure for the training data and for the estimated values that are derived the model.
        pylab.figure()
        pylab.plot(x, y, 'b.', label = 'Data points')
        pylab.plot(x, est_y, 'r-', label = 'Model')
        pylab.xlabel('years')
        pylab.ylabel('degrees Celsius')
        r2_se_str = '\nwith R Squared Value of {}'.format(r2)
        # Calculates the standard error over the slope of the model.
        if len(model) == 2:
            se = se_over_slope(x, y, est_y, model)
            r2_se_str += '\nand a Standard Error Over Slope Ratio of {}'.format(se)
        pylab.title('Predicated Temperatures for Model with a Degree of {}'.format(len(model) - 1) + r2_se_str)
        pylab.legend(loc = 'best')
        pylab.show()
         
def gen_cities_avg(climate, multi_cities, years):
    """
    Compute the average annual temperature over multiple cities.

    Args:
        climate: instance of Climate
        multi_cities: the names of cities we want to average over (list of str)
        years: the range of years of the yearly averaged temperature (list of
            int)

    Returns:
        a pylab 1-d array of floats with length = len(years). Each element in
        this array corresponds to the average annual temperature over the given
        cities for a given year.
    """
    temps = []
    city_avg_temp = 0
    for year in years:
        for city in multi_cities:
            city_daily_temps = climate.get_yearly_temp(city, year)
            city_avg_temp += ((city_daily_temps.sum()) / len(city_daily_temps))
        temps.append(city_avg_temp / len(multi_cities)) 
        city_avg_temp = 0
    return pylab.array(temps)

def moving_average(y, window_length):
    """
    Compute the moving average of y with specified window length.

    Args:
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        window_length: an integer indicating the window length for computing
            moving average

    Returns:
        an 1-d pylab array with the same length as y storing moving average of
        y-coordinates of the N sample points
    """
    total = 0
    counter = 0
    # move_avgs contains all the moving averages that are calculated from y.
    move_avgs = []
    for i in range(len(y)):
        total = 0
        counter = 0
        for j in range(window_length):
            index = i - j
            if index >= 0:    
                total += y[index]
                counter += 1
        avg = total / counter
        move_avgs.append(avg)
    return pylab.array(move_avgs)

def rmse(y, estimated):
    """
    Calculate the root mean square error term.

    Args:
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d pylab array of values estimated by the regression
            model

    Returns:
        a float for the root mean square error term
    """
    estimate_error = ((y - estimated)**2).sum()
    return pylab.sqrt(estimate_error / len(y))

def gen_std_devs(climate, multi_cities, years):
    """
    For each year in years, compute the standard deviation over the averaged yearly
    temperatures for each city in multi_cities. 

    Args:
        climate: instance of Climate
        multi_cities: the names of cities we want to use in our std dev calculation (list of str)
        years: the range of years to calculate standard deviation for (list of int)

    Returns:
        a pylab 1-d array of floats with length = len(years). Each element in
        this array corresponds to the standard deviation of the average annual 
        city temperatures for the given cities in a given year.
    """
    total_daily_temps = []
    stdevs = []
    error = 0  
    for year in years:
        # Creates a pylab array len(days in the year) with 0s as it's values.
        daily_temps = climate.get_yearly_temp(multi_cities[0], year)
        for i in range(len(daily_temps)):
            total_daily_temps.append(float(0))
        total_daily_temps = pylab.array(total_daily_temps)    
        # Averages the daily temperatures accross all cities.
        for city in multi_cities:
            total_daily_temps += climate.get_yearly_temp(city, year)
        total_daily_temps = total_daily_temps / len(multi_cities)    
        # Calculates the mean temperature for a given year across all cities.
        mean = total_daily_temps.sum() / len(total_daily_temps)
        for temp in total_daily_temps:
            error += (temp - mean)**2 # Calculates the error from the mean in each data point.
        stdevs.append(pylab.sqrt(error / len(total_daily_temps)))
        total_daily_temps = []
        error = 0               
    return pylab.array(stdevs)

def evaluate_models_on_testing(x, y, models):
    """
    For each regression model, compute the RMSE for this model and plot the
    test data along with the model's estimation.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (aka model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        RMSE of your model evaluated on the given data points. 

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a pylab array storing the coefficients of
            a polynomial.

    Returns:
        None
    """
    for model in models:
        # Calculates estimated y values(temps in degrees Celsius) from the models polynomial coefficients.
        est_y = pylab.polyval(model, x)
        error = rmse(y, est_y)
        # Creates a figure for the training data and for the estimated values that are derived the model.
        pylab.figure()
        pylab.plot(x, y, 'b.', label = 'Data points')
        pylab.plot(x, est_y, 'r-', label = 'Model')
        pylab.xlabel('years')
        pylab.ylabel('degrees Celsius')
        pylab.title('Predicated Temperatures for Model with a Degree of {}'.format(len(model) - 1) +
                    '\nwith RMSE value of {}'.format(error))
        pylab.legend(loc = 'best')
        pylab.show()
    

if __name__ == '__main__':
    # # Part A.4
    # climate = Climate('data.csv')
    # # x vals are years and y vals are daily temperatures.
    # x_vals, y_vals = [], [] 
    # for year in TRAINING_INTERVAL:
    #     x_vals.append(year)
    #     y_vals.append(climate.get_daily_temp('NEW YORK', 1, 10, year))
    # x_vals = pylab.array(x_vals)
    # y_vals = pylab.array(y_vals)
    # models = generate_models(x_vals, y_vals, [1])
    # evaluate_models_on_training(x_vals, y_vals, models)
    # # x vals are years and y vals are yearly temperatures.
    # y_vals = []
    # for year in TRAINING_INTERVAL:
    #     temps = climate.get_yearly_temp('NEW YORK', year)
    #     annual_temp = (temps.sum()) / len(temps)
    #     y_vals.append(annual_temp)
    # y_vals = pylab.array(y_vals)
    # models = generate_models(x_vals, y_vals, [1])
    # evaluate_models_on_training(x_vals, y_vals, models)    
    
    # # Part B
    # climate = Climate('data.csv')
    # # x_vals are years
    # x_vals = []
    # for i in TRAINING_INTERVAL:
    #     x_vals.append(i)
    # x_vals = pylab.array(x_vals)
    # # y_vals is a 1-d pylab array and it's values are the average yearly temperature accross multiple cities. 
    # y_vals = gen_cities_avg(climate, CITIES, TRAINING_INTERVAL) 
    # models = generate_models(x_vals, y_vals, [1])
    # evaluate_models_on_training(x_vals, y_vals, models)
    
    # # Part C
    # climate = Climate('data.csv')
    # # x_vals are years
    # x_vals = []
    # for i in TRAINING_INTERVAL:
    #     x_vals.append(i)
    # x_vals = pylab.array(x_vals)
    # # y_vals is a 1-d pylab array and it's values are the average yearly temperature accross multiple cities. 
    # y_vals = gen_cities_avg(climate, CITIES, TRAINING_INTERVAL) 
    # moving_avgs = moving_average(y_vals, 5) # Calculates the 5 year moving average for each year in y_vals.
    # models = generate_models(x_vals, moving_avgs, [1])
    # evaluate_models_on_training(x_vals, moving_avgs, models)

    # Part D.2
    # climate = Climate('data.csv')
    # # x_vals are years
    # x_vals = []
    # for i in TRAINING_INTERVAL:
    #     x_vals.append(i)
    # x_vals = pylab.array(x_vals)
    # # y_vals is a 1-d pylab array and it's values are the average yearly temperature accross multiple cities. 
    # y_vals = gen_cities_avg(climate, CITIES, TRAINING_INTERVAL) 
    # training_moving_avgs = moving_average(y_vals, 5) # Calculates the 5 year moving average for each year in the training interval.
    # models = generate_models(x_vals, training_moving_avgs, [1, 2, 20])
    # evaluate_models_on_training(x_vals, training_moving_avgs, models)
    # # x_vals are years
    # x_vals = []
    # for i in TESTING_INTERVAL:
    #     x_vals.append(i)
    # x_vals = pylab.array(x_vals)    
    # # y_vals is a 1-d pylab array and it's values are the average yearly temperature accross multiple cities. 
    # y_vals = gen_cities_avg(climate, CITIES, TESTING_INTERVAL) 
    # testing_moving_avgs = moving_average(y_vals, 5) # Calculates the 5 year moving average for each year in the testing interval.
    # evaluate_models_on_testing(x_vals, testing_moving_avgs, models)
    # Part E
    # climate = Climate('data.csv')
    # # x_vals are years
    # x_vals = []
    # for i in TRAINING_INTERVAL:
    #     x_vals.append(i)
    # x_vals = pylab.array(x_vals)
    # std_devs = gen_std_devs(climate, CITIES, TRAINING_INTERVAL)
    # moving_avgs = moving_average(std_devs, 5)
    # models = generate_models(x_vals, moving_avgs, [1])
    # evaluate_models_on_training(x_vals, moving_avgs, models)
    pass