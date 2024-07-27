import numpy as np
import matplotlib.pyplot as plt
from EventDisplay import EventDisplay


#1) make an instance of the event display class
eventDisplay = EventDisplay() 
#2) start by loading in the WCTE_mPMT_image_positions.npz file
eventDisplay.load_mPMT_positions("WCTE_mPMT_image_positions.npz")

#3) load the data to plot
#plot the geometry information to check that the event display is orientated correctly 
# data = np.load("wcsim_pi+_Beam_400MeV_30cm_0000geo.npz", allow_pickle=True)
# pmt_id = data["tube_no"]-1
# data_to_plot = pmt_id

#or plot some simulated event
eventID =0
data = np.load("wcsim_pi+_Beam_300MeV_25cm_0000.npz", allow_pickle=True)
pmt_id = data["digi_hit_pmt"][eventID]
data_to_plot = data["digi_hit_charge"][eventID]

#4) process the data from the numpy format into the correct format for the plotting software
data_to_plot = eventDisplay.process_data(pmt_id,data_to_plot)
#5) plot the data in the event display
eventDisplay.CNN_plot_data_2d(data_to_plot,style ="dark_background")
plt.show(block=False)
input()
plt.close()
