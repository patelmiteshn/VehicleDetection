from moviepy.editor import VideoFileClip

from image_processing_pipeline import *

# Input and OutputFile Location
infile = 'project_video.mp4'
outfile = 'project_video_output.mp4'




 # Load and process the video
original = VideoFileClip(infile)
processed = original.fl_image(find_cars)
processed.write_videofile(outfile, audio=False)