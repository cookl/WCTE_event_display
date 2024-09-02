import numpy as np
import matplotlib.pyplot as plt
from EventDisplay import EventDisplay
import matplotlib.colors as colors


#1) make an instance of the event display class
eventDisplay = EventDisplay() 
#2) start by loading in the CSV file for how the mPMTs are mapped to 2d event display
#note that the ordering of the mPMTs must match the datasource e.g. WCSim convention or WCTE convention
#different files can unmap the mPMTs differently
#event display with x axis coming out of page
# eventDisplay.load_mPMT_positions('mPMT_locations_WCSim_container_centreX.csv')
#event display with z axis coming out of page
eventDisplay.load_mPMT_positions('mPMT_locations_WCSim_container_centreZ.csv')
# eventDisplay.load_mPMT_positions('mPMT_2D_projection_angles.csv')

#mask out mPMT slots - nb this will only affect the event display the data loading is unchanged
#WCTE slot numbering
# eventDisplay.mask_mPMTs([45,77,79,27,32,85,91,99,12,14,16,18])
#WCSim container numbering
eventDisplay.mask_mPMTs([20,73,38,49,55,65,67,33,71,92,101,95])

# #3) load the data to plot
# plot some simulated event
eventID =0
data = np.load("test_data/wcsim_pi+_Beam_600MeV_30cm_0000.npz", allow_pickle=True)
pmt_id = data["digi_hit_pmt"][eventID]
data_to_plot = data["digi_hit_charge"][eventID]
#process data has a sum_data option set to true by default, 
# if set to false it will only plot the smallest quantity for each PMT e.g. time 
data_to_plot = eventDisplay.process_data(pmt_id,data_to_plot)
eventDisplay.plotEventDisplay(data_to_plot,color_norm=colors.Normalize(), style= "dark_background")
plt.show(block=False)
plt.savefig("Event_display.png")
input()
plt.close()

# #3) Or plot geometry data to check the event display mapping is orientated correctly
# #plot the geometry information to check that the event display is orientated correctly 

data = np.load("test_data/wcsim_pi+_Beam_400MeV_30cm_0000geo.npz", allow_pickle=True)
pmt_id = data["tube_no"]-1

for i, axis in enumerate(["x","y","z"]):
    data_to_plot = eventDisplay.process_data(pmt_id,data["position"][:,i])
    eventDisplay.plotEventDisplay(data_to_plot,color_norm=colors.Normalize(), style= "dark_background", show_zero=True)
    plt.show(block=False)
    plt.savefig("PMT_"+axis+"_position.png")
    input()
    plt.close()
