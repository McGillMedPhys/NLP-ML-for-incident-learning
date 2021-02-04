# -*- coding: utf-8 -*-
"""
This is an NLP program written based of the Notebook file created by Hua to predit different
classes of the incident learning description.
This python file is reading the databases and mapping and combining the two datasets. 
The file is split into multiple small ones because it takes long to execute otherwise and MacBook crashes
"""

import csv
import pandas



#-------- MUHC dataset -------------#
# Import original dataset
muhc = pandas.read_csv('../0_MUHC_data.csv', delimiter=',', encoding='latin-1').fillna('') # fillna function replaces NaN with empty string

# Create a temporary DataFrame of only the relevant columns.
temp = muhc[['event_type']].copy()
temp['ID'] = muhc['incident_id']
temp['Incident Description'] = muhc['descriptor'] + ' . ' + muhc['incident_description'] # id for incident description
temp['Process Step'] = muhc['process_step_occurred']
temp['Problem Type'] = muhc['problem_type']
temp['Contributing Factors'] = muhc['contributing_factors']
temp['Overall Severity'] = muhc['acute_medical_harm']
muhc = temp[['ID', 'Process Step', 'Problem Type', 'Contributing Factors', 'Overall Severity', 'Incident Description']]

#-------- CIHI dataset -------------#
# Import original dataset
cihi = pandas.read_csv('../0_CIHI_data.csv', delimiter=',', encoding='latin-1').fillna('')

# Create a temporary DataFrame of only the relevant columns.
temp = cihi[['Frequency']].copy()
temp['ID'] = cihi['RT NSIR Case ID']
temp['Incident Description'] = cihi['Incident Description']
temp['Process Step'] = cihi['Process Step Where Incident Occurred']
temp['Problem Type'] = cihi['Problem Type - Primary']
temp['Contributing Factors'] = cihi['Contributing Factor List']
temp['Overall Severity'] = cihi['Overall Severity']
cihi = temp[['ID', 'Process Step', 'Problem Type', 'Contributing Factors', 'Overall Severity', 'Incident Description']]





#------------ Mapping CIHI data to MUHC -------------#

# Definition of our mapping function
def map_value(mapping, old_value):
    try:
        return mapping[old_value]
    except Exception:
        return old_value
    
##### Process Step ########
# This step gets the list of all unique labels in the list and sort them 
set(muhc['Process Step']) 
set(cihi['Process Step'])    

ps_mapping = {}   # initiating a library

'''
The following script maps the process steps in the CIHI list into the ones correspoinding to the MUHC list.
Note that the mapping is performed based on the NSIR-RT mapping document.
'''
# From CIHI to MUHC
ps_mapping['Contouring and planning'] = 'Treatment planning'
ps_mapping['Imaging for treatment planning'] = 'Imaging for radiotherapy planning'
ps_mapping['Interventional procedure for planning and/or delivery'] = 'Patient assessment or consultation'
ps_mapping['Not Applicable'] = ''
ps_mapping['On-treatment quality assurance'] = 'On-treatment quality management'
ps_mapping['Patient assessment/consultation (Retired Value)'] = 'Patient assessment or consultation'
ps_mapping['Patient medical consultation and physician assessment'] = 'Patient assessment or consultation'
ps_mapping['Radiation treatment prescription scheduling'] = 'Patient assessment or consultation'
ps_mapping['Pre-treatment quality assurance'] = 'Pre-treatment review and verification'
ps_mapping['Post-treatment completion '] = 'Post-treatment completion'

muhc['Process Step'] = muhc['Process Step'].apply(lambda x:map_value(ps_mapping, x))
cihi['Process Step'] = cihi['Process Step'].apply(lambda x:map_value(ps_mapping, x))


######### Problem Type ###############
set(muhc['Problem Type'])
set(cihi['Problem Type'])

pt_mapping = {}

'''
The the problem type labels that are not directly relatable to the labels in the MUHC list are marked as other.
'''
# From CIHI to MUHC
pt_mapping['Bleeding'] = 'Other'
pt_mapping['Excess imaging dose '] = 'Other'
pt_mapping['Failure to perform on-treatment imaging per instructions'] = 'Other'
pt_mapping['Fall or other patient injury or medical condition'] = 'Fall or other accident'
pt_mapping['Inadequate coordination of combined modality care'] = 'Combined modality treatment scheduling error'
pt_mapping['Inappropriate or poorly informed decision to treat or plan'] = 'Other'
pt_mapping['Infection'] = 'Other'
pt_mapping['Interventional procedure error (Retired value)'] = 'Interventional procedure error'
pt_mapping['Radiation therapy scheduling error'] = 'Radiation treatment scheduling error'
pt_mapping['Systematic hardware/software (including dose├â┬â├é┬é├â┬é├é┬┐volume) error'] = 'Hardware/Software'
pt_mapping['Treatment not delivered: personnel/hardware/software failure'] = 'Other'
pt_mapping['Treatment plan (isodose distribution) unacceptable'] = 'Other'
pt_mapping['Treatment plan acceptable but not physically deliverable'] = 'Other'
pt_mapping['Untimely access to medical care or radiotherapy'] = 'Other'
pt_mapping['Wrong anatomical site (excluding laterality)'] = 'Wrong anatomical site'
pt_mapping['Wrong patient position, setup point or shift'] = 'Wrong patient position'
pt_mapping['Wrong plan dose (Retired value)'] = 'Other'
pt_mapping['Wrong planning margins'] = 'Wrong target or OAR contours, or wrong planning margins'
pt_mapping['Wrong prescription dose fractionation or calculation error'] = 'Calculation error'
pt_mapping['Wrong side (laterality)'] = 'Wrong patient position'
pt_mapping['Wrong target or OAR contours'] = 'Wrong target or OAR contours, or wrong planning margins'
pt_mapping['Wrong target or OAR contours or wrong planning (Retired Value)'] = 'Wrong target or OAR contours, or wrong planning margins'
pt_mapping['Wrong, missing, mislabeled or damaged treatment accessories'] = 'Wrong treatment accessories'
pt_mapping['Patient movement during simulation or treatment'] = 'Other'

muhc['Problem Type'] = muhc['Problem Type'].apply(lambda x:map_value(pt_mapping, x))
cihi['Problem Type'] = cihi['Problem Type'].apply(lambda x:map_value(pt_mapping, x))


########### Contributing Factors #############
'''
The contributing factor labels are little different from the process step and problem type labels.
Here some lables are groups of multiple labels seperated by comma and ambersand symbols. Hence the processing of these is different.
'''
def map_muhc_cfs(cfs):
    mapped_values = []
    for cf in cfs.split('&'):
        try:
            mapped_values.append(cf_mapping[cf.strip()])
        except Exception:
            mapped_values.append(cf.strip().replace(',', '').replace('\'', ''))
    return '|'.join(mapped_values)

def map_cihi_cfs(cfs):
    cf_list = []
    temp = cfs.replace('Equipment software or hardware design, including human factors design, inadequate', 'Equipment software or hardware design including human factors design inadequate')
    temp = temp.replace('Equipment software or hardware commissioning, calibration or acceptance testing inadequate', 'Equipment software or hardware commissioning calibration or acceptance testing inadequate')
    temp = temp.replace('Patient or family member medical condition, preference or behaviour', 'Patient or family member medical condition preference or behaviour')
    for cf in temp.split(','):
        cf_list.append(cf.strip())   
    mapped_cfs = []
    for cf in cf_list:
        try:
            mapped_cfs.append(cf_mapping[cf.strip()])
        except Exception:
            mapped_cfs.append(cf.strip())
    return '|'.join(mapped_cfs)


# muhc_cfs = []
# for cfs in set(muhc['Contributing Factors']):  
#     for cf in cfs.split('&'):   # Some labels in the list are a group of more than one label connected with an "&". This step seperate each label from such groups
#         muhc_cfs.append(cf.strip())

# cihi_cfs = []
# for cfs in set(cihi['Contributing Factors']):
#     # Some labels have comma within it. The replace function is used to replace 3 such labels with their comma less versions in the format: replace (old version, new version)
#     if ("Equipment software or hardware design, including human factors design, inadequate" in cfs): print (cfs)
#     temp = cfs.replace("Equipment software or hardware design, including human factors design, inadequate", "Equipment software or hardware design including human factors design inadequate")
#     temp = temp.replace("Equipment software or hardware commissioning, calibration or acceptance testing inadequate", "Equipment software or hardware commissioning calibration or acceptance testing inadequate")
#     temp = temp.replace("Patient or family member medical condition, preference or behaviour ", "Patient or family member medical condition preference or behaviour")
#     for cf in temp.split(','):  #similar to the MUHC data set, multiple labels are grouped together with comma here. This step seperates them.
#         cihi_cfs.append(cf.strip())
        
# set(muhc_cfs)
# set(cihi_cfs)

cf_mapping = {}

'''
Mapped to the most appropriate labels
'''
#From CIHI to MUHC
cf_mapping['Change management']= 'Other'
cf_mapping['Communication or documentation inadequate (patient specific)']= 'Communication inappropriate or misdirected'
cf_mapping['Distraction or diversions involving staff']=  'Loss of attention'
cf_mapping['Equipment quality assurance and/or maintenance inadequate']= 'Materials tools or equipment inadequate or insufficient'
cf_mapping['Equipment software or hardware commissioning calibration or acceptance testing inadequate']= 'Commissioning or acceptance testing inadequate'
cf_mapping['Equipment software or hardware design including human factors design inadequate']= 'Physical environment inadequate'
cf_mapping['Expectation bias involving staff']= 'Expectation bias'
cf_mapping['Failure to identify potential risks']= 'Failure to recognize a hazard'
cf_mapping['Handoffs inadequate']= 'Other'
cf_mapping['Organizational and/or workspace resources inadequate (excluding human resources)']= 'Capital resources inadequate'
cf_mapping['Patient education inadequate']= 'Patient related circumstances'
cf_mapping['Patient or family member medical condition preference or behaviour']= 'Patient related circumstances'
cf_mapping['Policies and/or procedures non-existent or inadequate']= 'Non-existent'
cf_mapping['Policies and/or procedures not followed']= 'Failure to select the correct rule'
cf_mapping['Staff behaviour']= 'Human behavior involving staff'
cf_mapping['Staff education or training inadequate']= 'Education or training inadequate'
cf_mapping['Unfamiliar treatment approach or radiation treatment technique']= 'Education or training inadequate'


muhc['Contributing Factors'] = muhc['Contributing Factors'].apply(lambda x:map_muhc_cfs(x))
cihi['Contributing Factors'] = cihi['Contributing Factors'].apply(lambda x:map_cihi_cfs(x))

# ############ Overall Severity ###############
# set(muhc['Overall Severity'])
# set(cihi['Overall Severity'])

# os_mapping = {}

# # From CIHI to MUHC
# os_mapping['Not Applicable'] = ''
# os_mapping['Severe'] = ''
# os_mapping['Moderate'] = ''

# muhc['Overall Severity'] = muhc['Overall Severity'].apply(lambda x:map_value(os_mapping, x))
# cihi['Overall Severity'] = cihi['Overall Severity'].apply(lambda x:map_value(os_mapping, x))


#---------------- Combining both datasets into one ---------------#
combined = pandas.concat([muhc, cihi])
combined.to_csv('../out/1_Combined.csv', encoding='utf-8', index=False, quoting=csv.QUOTE_NONNUMERIC)
