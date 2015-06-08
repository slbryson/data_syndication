import pandas as pd
import numpy as np
import re
import matplotlib as plt

import csv

def write_top_sectors(dm,top_count_sector):
	# Define and write top offending sectors to csv
    filename = 'output/top_cdma_drop_list-fullmarket.txt'
    dm3 = dm.loc[dm['Carrier']>top_count_sector]
    #dm3 = dm2[['Cell','Sector', 'Carrier']].groupby([dm2['Cell'], dm2['Sector'], dm2['Carrier']]).agg(len)
    dm3.sort(columns='Carrier',ascending = False)


    print type(dm3), '\nNumber of Sectors', len(dm3), dm3.columns

    if True:
        dm3.to_csv('output/top_cdma_drop_list-fullmarket.txt')
        print "Succcessfully wrote out ", len(dm3), ' items to ',filename

    #Let's attempt some clean up
    del dm3



#Careful this is called within agg_by
def norm_cdf(mean=0.0, std=1.0, min=0, max=500):
    import scipy as sp
    from scipy import stats
    import matplotlib as plt
    x = sp.linspace(min, max, 50)
    # CDF at these values
    y = stats.norm.cdf(x, loc=mean, scale=std)



def check_band(df1):

    CDMA = df1['Ending Band Class'].str.contains('CDMA').sum()
    PCS =df1['Ending Band Class'].str.contains('PCS').sum()
    Unknown =df1['Ending Band Class'].str.contains('Unknow').sum()

    CDMA = CDMA.astype(float)
    PCS = PCS.astype(float)
    Unknown = Unknown.astype(float)

    print "CDMA band = %2.f percent"%((100)*CDMA/len(df1))
    print "PCS band = %2.f percent"%((100)*PCS/len(df1))
    print "Unknown band = %2.f percent"%((100)*Unknown/len(df1))
    print "Total data set size =", len(df1)
    return CDMA, PCS, Unknown, len(df1)



def check_drop(df3):
    df3[['Analysis Type']] = df3[['Analysis Type']].astype(str)
    datasize = len(df3)

    two_way = df3['Analysis Type'].str.contains('TwoWay').sum()
    weak_active =df3['Analysis Type'].str.contains('WeakActive').sum()
    missing =df3['Analysis Type'].str.contains('Missing').sum()

    two_way = two_way.astype(float)
    weak_active = weak_active.astype(float)
    missing = missing.astype(float)
    print "Weak Active Set = %2.1f perecent"%((100)* weak_active/datasize), "\nTwo Way = %2.1f percent "%((100)*two_way/datasize)
    print "Missing PN = %2.1f perecent"%((100)* missing/datasize)
    print "Data set size = ", datasize
    return two_way, weak_active, missing, datasize


def make_mean_carrier(dm):
    #Create the statistics on a data set sorted also by sector.
	mean_count = dm['Carrier'].mean()
	if mean_count.dtype == float and np.isnan(mean_count):
	    mean_count = 0
	std_count = dm['Carrier'].std()
	if std_count.dtype == float and np.isnan(std_count):
	    std_count = 0
	max_count = dm['Carrier'].max()
	min_count = dm['Carrier'].min()
	top_count_sector = mean_count + std_count

	bad_sectors = dm.loc[dm['Carrier']>top_count_sector]

	sector_count = float(len(dm))
	try: 
	    bad_sectors = float(bad_sectors)
	except:
	    bad_sectors =0
	#Dropped Call Failure reason
	#Call Final Class qualifier
     
        print '++++++++++++++++++++\n','Total Sector Count = %4d'%(sector_count)
        print '\nNumber of Top Offending Sectors = %4d '%bad_sectors
        print '%4.2f percent'%((100)*(bad_sectors/sector_count))
        # Find the cutoff statistics
        print '\nMean =%4d'%(mean_count), 'Standard deviation ',std_count, 'Mean + 1 sigma = %4d'%(top_count_sector)
        if False:
            fig =plt.figure()
            plt.clf()
            bin_tick =np.arange(top_count_sector,max_count,50, dtype=int)

            #dm2.hist()
            #dm2.plot(x=['ECP','Cell'], y ='Carrier',kind='line')
            #dm.plot(x=['ECP','Cell'], y='Carrier' )
            #plt.subplots(2,2)
            ax1 = fig.add_subplot(2,2,1)

            _ = ax1.plot(dm['Carrier'].values, drawstyle='steps-post', label ='steps-post')
            ax2 = fig.add_subplot(2,2,3)

            bin_tick = np.arange(min_count,max_count,50, dtype=int)

            _ = ax2.hist(dm['Carrier'],bins=bin_tick)
            ax3 = fig.add_subplot(2,2,4)
 	    norm_cdf(68, std_count, min_count, max_count)
	return top_count_sector



def make_mean(dm_cell):
    # Computer the mean of the full monty (df2)
    #df2.describe()

    mean_count = dm_cell.values.mean()
    std_count = dm_cell.values.std()
    max_count = dm_cell.values.max()
    min_count = dm_cell.values.min()
    top_count = mean_count + std_count

    #Dropped Call Failure reason
    #Call Final Class qualifier
    top_off=float(len(dm_cell.loc[dm_cell['Cell']>top_count]))
    total_cell = float(len(dm_cell))

    print '\nTotal Number of Cells in Filtered List = ', len(dm_cell),      '\nNumber of cells in top offenders = ', len(dm_cell.loc[dm_cell['Cell']>top_count])
    print '%4.2f percent'%((100)*(top_off/total_cell))
    # Find the cutoff statistics
    print '\nMean =', mean_count, 'Standard deviation ',std_count, 'Mean + 1 sigma = %4d'%(mean_count+std_count)
    return top_count
 



def  agg_by(df3):
    #Now in a separate cell we compute the different groups using the filtered copy we made.
    if True:
        #dm_sector = df3[['ECP', 'Cell','Sector']].groupby([df3['ECP'], df3['Cell'], df3['Sector']]).agg(len)
        # dm_cell is still using the method that leaves a multi-level index
        dm_cell = df3[['ECP', 'Cell']].groupby([df3['ECP'], df3['Cell']]).agg(len)
        dm_cell.sort(columns='Cell',ascending =False,inplace = True)
        #Here we use a method to unravel the multi-level index from the start.
        dm = df3[['ECP', 'Cell','Sector','Carrier']].groupby([df3['ECP'], df3['Cell'],df3['Sector'],df3['Carrier']]).agg(len)
        dm.sort(columns='Carrier', ascending = False, inplace =True)


         
        #print '\ndm_sector info \n', dm.info(), '\ndm_cell info \n',dm_cell.info()
        print '\nNumber of items in dm_sector = ', len(dm)
        print '\nNumber of items in dm_cell = ', len(dm_cell)

    else:
        dm = df2[['ECP', 'Cell','Call Final Class qualifier']].groupby([df2['ECP'], df2['Cell'],df2['Call Final Class qualifier']]).agg(len)
        dm.sort(columns='Cell',ascending = False, inplace =True)
        print 'df2 info this is unfiltered ', df2.info()
        print '\n\nNow the grouped series dm ',dm.info()
    return dm, dm_cell







def get_dataframe(inputFile,maxReadRows):

    import pandas as pd
    import numpy as np
    import re
    import csv

    lst_myFieldNames = ['ECP', 'Cell', 'Sector', 'Carrier', 'Date', 'Hour', 
                       'Dropped Call Timestamp', 'Call Final Class qualifier', 
                       'Secondary Call Final Class qualifier', 'Analysis Type', 
                       'Analysis Reason', 'Mobile Vendor', 'Mobile serial number', 
                       'Ending Band Class', 'Average FL Power (Watts)', 
                       'Average RL RSSI rise (dB)']
    lst_Rows = []
    lst_myFieldNumbers = []
    rowCount = 0
    with open(inputFile) as csvFile:
        csvReader = csv.reader(csvFile, dialect = 'excel')
        firstLine = next(csvReader)
        i = 0
        for field in firstLine:
            if field in lst_myFieldNames:
                lst_myFieldNumbers.append(i)
            i += 1

        for row in csvReader:
            #lst_Row = [row[i] for i in lst_myFieldNumbers]
            lst_Row = []
            for i in lst_myFieldNumbers:
                if len(row[i]) == 0:
                    lst_Row.append(np.nan)
                else:
                    lst_Row.append(row[i])
            lst_Rows.append(lst_Row)
            rowCount += 1
            if rowCount == maxReadRows:
                break

    df1 = pd.DataFrame(data = lst_Rows,
                       columns = lst_myFieldNames,
                       dtype = unicode)
    return df1



# Create a count summed by Drop Reason
def agg_by_reason(dm2):
    
    dm3 = dm2[['ECP', 'Cell','Sector','Analysis Type']].groupby([dm2['ECP'], dm2['Cell'], dm2['Sector'],dm2['Analysis Type']]).agg(len)
    print type(dm3), '\nNumber Rows for the group by Sector ', len(dm3)

    dm3.sort(columns='Cell',ascending = False)

    #Print results if Data is correct
    if True:
        dm3.to_csv('output/top_cdma_drop_reason.txt')



def agg_by_bandclass(df3):
    
    # Create a count to look for the cells with the most drops
    #this is obvsioulsy broken.  Likely dm2 is not the right starting point to sort.
    #Same as above I seem to be off by a column somewhere.
    dm3 = df3[['Cell','Sector','Ending Band Class']].groupby([df3['Cell'], df3['Sector'],df3['Ending Band Class']]).agg(len)

    print type(dm3), '\nEnding Band Class', len(dm3)

    dm3.sort(columns='Cell',ascending = False)
    if True:
        dm3.to_csv('output/top_cdma_drop_BC.txt')
    del dm3



def agg_by_RSSI(df3):
    dm3 = df3[['Cell','Sector', 'Carrier','Average RL RSSI rise (dB)']].groupby([df3['Cell'],                          df3['Sector'], df3['Carrier'], df3['Average RL RSSI rise (dB)']]).agg(len)
    print type(dm3), '\nNumber of Sector', len(dm3)
    if True:
        dm3.to_csv('output/top_cdma_drop_RSSI.txt')
    del dm3

