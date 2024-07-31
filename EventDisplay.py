import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import matplotlib.colors as colors
import copy
from contextlib import nullcontext

class EventDisplay:
  
    def load_mPMT_positions(self,fileName):   
        #load the mpmt_positions file and create a series of variables used throughout the 
        self.mpmt_positions = np.load(fileName)['mpmt_image_positions']
        data_size = np.max(self.mpmt_positions, axis=0) + 1
        #max of x and y separately

        #for each row (x point which has a position in every point along the barrel )
        barrel_rows = [row for row in range(data_size[0]) if np.count_nonzero(self.mpmt_positions[:, 0] == row) == data_size[1]]
        #endcap starts at smallest barrel row 
        endcap_size = np.min(barrel_rows)

        #creates an index object that in the second last dimension
        #picks out the barrel from the data format 
        self.barrel = np.s_[..., endcap_size:np.max(barrel_rows) + 1, :]
        self.image_height, self.image_width = np.max(self.mpmt_positions, axis=0) + 1

        #different mapping of PMTs into mPMT
        self.barrel_mpmt_map = np.array([0, 4, 5, 6, 1, 2, 3, 13, 14, 15, 16, 17, 18, 7, 8, 9, 10, 11, 12])

    def process_data(self,pmts, pmt_data):
        #"""Returns image-like event data array (channels, rows, columns) from arrays of PMT IDs and data at the PMTs."""
        # Ensure the data is a 2D array with first dimension being the number of pmts
        
        pmt_data = np.atleast_1d(pmt_data)
        # pmt_data = pmt_data.reshape(pmt_data.shape[0], -1)

        PMTS_PER_MPMT =19
        mpmts = pmts // PMTS_PER_MPMT #mPMT ID for each PMT
        print(mpmts)
        channels = pmts % PMTS_PER_MPMT #channel ID for each PMT
        #for each channel get the row and column of the mPMT
        rows = self.mpmt_positions[mpmts, 0] 
        cols = self.mpmt_positions[mpmts, 1] #row and column of each hit point

        # data = np.zeros((PMTS_PER_MPMT, pmt_data.shape[1], image_height, image_width), dtype=np.float32)
        data = np.zeros((PMTS_PER_MPMT, self.image_height, self.image_width), dtype=np.float32)
        # data[channels, :, rows, cols] = pmt_data
        data[channels, rows, cols] = pmt_data

        # fix indexing of barrel PMTs in mPMT modules to match that of endcaps in the projection to 2D
        
        data[self.barrel] = data[self.barrel_mpmt_map][self.barrel]
        return data

    def channel_position_offset(self, channel):
        """
        Calculate offset in plotting coordinates for channel of PMT within mPMT.

        Parameters
        ----------
        channel: array_like of float
            array of channel IDs or PMT IDs
        use_new_convention: bool
            use newer convention for the channel mapping of PMTs in the mPMT (starts with central PMT, then middle ring then
            outer ring) as opposed to old convention (starts with outer ring, then middle ring, then central PMT).

        Returns
        -------
        np.ndarray:
            Array of (y, x) coordinate offsets
        """
        channel = channel % 19
        # if use_new_convention:
        theta = (channel > 6)*2*np.pi*(19-channel)/12 + ((channel > 0) & (channel <= 6))*2*np.pi*(7-channel)/6 + np.pi/2
        
        radius = 0.2*(channel > 0) + 0.2*(channel > 6)
        # else:
        #     theta = (channel < 12)*2*np.pi*channel/12 + ((channel >= 12) & (channel < 18))*2*np.pi*(channel-12)/6
        #     radius = 0.2*(channel < 18) + 0.2*(channel < 12)
        position = np.column_stack((radius*np.sin(theta), radius*np.cos(theta)))
        return position

    def coordinates_from_data(self, data):
        """
        Calculate plotting coordinates for each element of CNN data, where each element of `data` contains a PMT's data.
        The actual values in `data` don't matter, it just takes the data tensor that has dimensions of
        (channel, image row, image column) and returns plotting coordinates for each element of the flattened data array.
        Plotting coordinates returned correspond to [x, y] coordinates for each PMT, including offsets for the PMT's
        position in the mPMT.

        Parameters
        ----------
        data: array_like
            Array of PMT data formatted for use in CNN, i.e. with dimensions of (channel, row, column)
        use_new_convention: bool
            use newer convention for the channel mapping of PMTs in the mPMT (starts with central PMT, then middle ring then
            outer ring) as opposed to old convention (starts with outer ring, then middle ring, then central PMT).

        Returns
        -------
        coordinates: np.ndarray
            Coordinates for plotting the data
        """
        indices = np.indices(data.shape)
        channels = indices[0].flatten()
        coordinates = indices[[2, 1]].reshape(2, -1).astype(np.float64).T
        coordinates += self.channel_position_offset(channels)
        
        #flip the y coordinate to account for the watchmal event display being upside down 
        coordinates[:, 1] = -1*coordinates[:, 1]
        coordinates[:, 0] = -1*coordinates[:, 0]
        return coordinates

    def CNN_plot_data_2d(self, data, channel=None, transforms=None, **kwargs):
        # use_new_mpmt_convention = True
        """
        Plots CNN mPMT data as a 2D event-display-like image.

        Parameters
        ----------
        data : array_like
            Array of PMT data formatted for use in CNN, i.e. with dimensions of (channels, x, y)
        channel : str
            Name of the channel to plot. By default, plots whatever is provided by the dataset.
        transforms : function or str or sequence of function or str, optional
            Transformation function, or the name of a method of the dataset, or a sequence of functions or method names
            to apply to the data, such as those used for augmentation.
        kwargs : optional
        Additional arguments to pass to `analysis.event_display.plot_event_2d`.
        Valid arguments are:
        fig_width : scalar, optional
            Width of the figure
        title : str, default: None
            Title of the plot
        style : str, optional
            matplotlib style
        color_label: str, default: "Charge"
            Label to print next to the color scale
        color_map : str or Colormap, default: plt.cm.plasma
            Color map to use when plotting the data
        color_norm : matplotlib.colors.Normalize, optional
            Normalization to apply to color scale, by default uses log scaling
        show_zero : bool, default: false
                If false, zero data is drawn as the background color

        Returns
        -------
        fig: matplotlib.figure.Figure
        ax: matplotlib.axes.Axes
        """
        rows = self.mpmt_positions[:, 0]
        columns = self.mpmt_positions[:, 1]
        data_nan = np.full_like(data, np.nan)  # fill an array with nan for positions where there's no actual PMTs
        data_nan[:, rows, columns] = data[:, rows, columns]  # replace the nans with the data where there is a PMT
        # if transforms is not None:
        #     data_nan = self.apply_transform(transforms, {"data": data_nan})["data"]
        # if channel is not None:
        #     data_nan = data_nan[self.channel_ranges[channel]]
        coordinates = self.coordinates_from_data(data_nan)  # coordinates corresponding to each element of the data array
        # also get the coordinates of the centre PMT (channel 18 or 0 depending on convention) where the actual mPMTs are (not nan)
        channel = 0 
        mpmt_coordinates = coordinates[((~np.isnan(data_nan)) & (np.indices(data_nan.shape)[0] == channel)).flatten()]
        return self.plot_event_2d(data_nan.flatten(), coordinates, mpmt_coordinates, **kwargs)
    
    def plot_event_2d(self,pmt_data, data_coordinates, pmt_coordinates, fig_width=None, title=None, style=None,
                    color_label=None, color_map=plt.cm.plasma, color_norm=colors.LogNorm(), show_zero=False):
        """
        Plots 2D event display from PMT data

        Parameters
        ----------
        pmt_data : array_like
            Data to be plotted (e.g. PMT charges)
        data_coordinates : array_like
            2D (x, y) locations of where to plot the data
        pmt_coordinates : array_like
            2D (x, y) locations for circles to draw where PMTs or mPMTs are present
        fig_width : scalar, optional
            Width of the figure
        title : str, default: None
            Title of the plot
        style : str, optional
            matplotlib style
        color_label: str, default: "Charge"
            Label to print next to the color scale
        color_map : str or Colormap, default: plt.cm.plasma
            Color map to use when plotting the data
        color_norm : matplotlib.colors.Normalize, optional
            Normalization to apply to color scale, by default uses log scaling
        show_zero : bool, default: false
            If false, zero data is drawn as the background color

        Returns
        -------
        fig: matplotlib.figure.Figure
        ax: matplotlib.axes.Axes
        """
        if not show_zero:
            pmt_data[pmt_data == 0] = np.nan
        color_map = copy.copy(color_map)
        if style == "dark_background":
            edge_color = '0.35'
            color_map.set_bad(color='black')
        else:
            edge_color = '0.85'
            color_map.set_bad(color='white')
        axis_ranges = np.ptp(pmt_coordinates, axis=0)
        if fig_width is None:
            fig_width = matplotlib.rcParams['figure.figsize'][0]
        scale = fig_width/20
        fig_size = (20*scale, 16*scale*axis_ranges[1]/axis_ranges[0])
        pmt_circles = [Circle((pos[0], pos[1]), radius=0.48) for pos in pmt_coordinates]
        
        with plt.style.context(style) if style else nullcontext():
            fig, ax = plt.subplots(figsize=fig_size)
            ax.set_aspect(1)
            ax.add_collection(PatchCollection(pmt_circles, facecolor='none', linewidths=1*scale, edgecolors=edge_color))
            pmts = ax.scatter(data_coordinates[:, 0], data_coordinates[:, 1], c=pmt_data.flatten(), s=7*scale*scale, cmap=color_map, norm=color_norm)
            ax_min = np.min(pmt_coordinates, axis=0) - 1
            ax_max = np.max(pmt_coordinates, axis=0) + 1
            ax.set_xlim([ax_min[0], ax_max[0]])
            ax.set_ylim([ax_min[1], ax_max[1]])
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            fig.colorbar(pmts, ax=ax, pad=0, label=color_label)
        if title is not None:
            ax.set_title(title)
        return fig, ax

 
#
# eventDisplay = EventDisplay() 
# eventDisplay.load_mPMT_positions("WCTE_mPMT_image_positions.npz")

# #plot the geometry information to check that the event display is orientated correctly 
# # data = np.load("wcsim_pi+_Beam_400MeV_30cm_0000geo.npz", allow_pickle=True)
# # pmt_id = data["tube_no"]-1
# # data_to_plot = pmt_id

# #or plot some simulated event
# eventID =0
# data = np.load("wcsim_pi+_Beam_300MeV_25cm_0000.npz", allow_pickle=True)
# pmt_id = data["digi_hit_pmt"][eventID]
# data_to_plot = data["digi_hit_charge"][eventID]


# data_to_plot = eventDisplay.process_data(pmt_id,data_to_plot)
# eventDisplay.CNN_plot_data_2d(data_to_plot,style ="dark_background")
# plt.show(block=False)
# input()
# plt.close()
