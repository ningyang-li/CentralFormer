There are two ways to load model weights.
<ul>
  <li><h4>Load weights with keras method, for example,</h4></li>
  <code>
    model.save_weights(path)    # path can be r"D:\...\weights.h5"
    model.load_weights(path)</code>
  <li><h4>Load weights with our's method which depends on the pickle package, for example,</h4></li>
  <code>
    network.save_weights(path)    # path can be r"D:\...\weights.pickle"
    network.load_weights(path)
    # You can try this method when your hdf5 package is not available.</code>
</ul>
