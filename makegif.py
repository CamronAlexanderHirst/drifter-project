import imageio
images = []
filenames = []
for i in range(40):
    filenames.append('./figures/'+str(i)+'.png')

for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('movie.gif', images, duration=0.2)
