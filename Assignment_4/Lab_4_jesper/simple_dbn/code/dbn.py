from util import *
from rbm import RestrictedBoltzmannMachine
import matplotlib.pyplot as plt
import sys
class DeepBeliefNet():

    '''
    For more details : Hinton, Osindero, Teh (2006). A fast learning algorithm for deep belief nets. https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf

    network          : [top] <---> [pen] ---> [hid] ---> [vis]
                               `-> [lbl]
    lbl : label
    top : top
    pen : penultimate
    hid : hidden
    vis : visible
    '''

    def __init__(self, sizes, image_size, n_labels, batch_size):

        """
        Args:
          sizes: Dictionary of layer names and dimensions
          image_size: Image dimension of data
          n_labels: Number of label categories
          batch_size: Size of mini-batch
        """

        self.rbm_stack = {

            'vis--hid' : RestrictedBoltzmannMachine(ndim_visible=sizes["vis"], ndim_hidden=sizes["hid"],
                                                    is_bottom=True, image_size=image_size, batch_size=batch_size),

            #'hid--pen' : RestrictedBoltzmannMachine(ndim_visible=sizes["hid"], ndim_hidden=sizes["pen"], batch_size=batch_size),

            'hid+lbl--top' : RestrictedBoltzmannMachine(ndim_visible=sizes["hid"]+sizes["lbl"], ndim_hidden=sizes["top"],
                                                        is_top=True, n_labels=n_labels, batch_size=batch_size)
        }

        self.sizes = sizes

        self.image_size = image_size

        self.batch_size = batch_size

        self.n_gibbs_recog = 15

        self.n_gibbs_gener = 200

        self.n_gibbs_wakesleep = 5

        self.print_period = 100

        self.n_labels = n_labels

        return

    def recognize(self,true_img,true_lbl):

        """Recognize/Classify the data into label categories and calculate the accuracy

        Args:
          true_imgs: visible data shaped (number of samples, size of visible layer)
          true_lbl: true labels shaped (number of samples, size of label layer). Used only for calculating accuracy, not driving the net
        """

        n_samples = true_img.shape[0]
        predicted_lbl = np.zeros((0, self.n_labels))
        mini_batch_size = 100
        result = []
        # vis = true_img # visible layer gets the image data
        # lbl = np.ones(true_lbl.shape)/10. # start the net by telling you know nothing about labels
        for mbs in range(0, true_img.shape[0], mini_batch_size):

            vis = true_img[mbs:mbs+mini_batch_size,:]
            img_gen = vis[0]

            lbl = np.ones((vis.shape[0],self.n_labels))/10.

            # [TODO TASK 4.2] fix the image data in the visible layer and drive the network bottom to top. In the top RBM, run alternating Gibbs sampling \
            # and read out the labels (replace pass below and 'predicted_lbl' to your predicted labels).
            # NOTE : inferring entire train/test set may require too much compute memory (depends on your system). In that case, divide into mini-batches.

            vis = self.rbm_stack['vis--hid'].get_h_given_v_dir(vis)[1]
            #vis = self.rbm_stack['hid--pen'].get_h_given_v_dir(vis)[1]

            vis = np.concatenate((vis, lbl), axis = 1) #Column concatenation
            #rbm = self.rbm_stack['pen+lbl--top'] #Topmost layer
            rbm = self.rbm_stack['hid+lbl--top']

            for _ in range(self.n_gibbs_recog): # Gibbs sampler in the top layer
                vis = rbm.get_v_given_h(rbm.get_h_given_v(vis)[1])[1] #Gibbs sampling.

            predicted_lbl = np.concatenate((predicted_lbl, vis[:,-self.n_labels:]), axis=0)
            """print(vis[0,-self.n_labels:])
            print(true_lbl[mbs])
            sys.stdout.flush()
            plt.imshow(img_gen.reshape(self.image_size))
            plt.show()"""
            #print(predicted_lbl)

            #if mbs % 1000 == 0:
            print(mbs)
            result.append(100.*np.mean(np.argmax(predicted_lbl,axis=1)==np.argmax(true_lbl[:mbs+mini_batch_size, :],axis=1)))
        print ("accuracy = %.2f%%"%(100.*np.mean(np.argmax(predicted_lbl,axis=1)==np.argmax(true_lbl[:mbs+mini_batch_size, :],axis=1))))
        plt.plot(result)
        plt.title("Recognition accuracy")
        plt.xlabel("Iterations (100)")
        plt.ylabel("Accuracy (%)")
        plt.show()
            #print ("accuracy = %.2f%%"%(100.*np.mean(np.argmax(predicted_lbl,axis=1)==np.argmax(true_lbl,axis=1))))

        return

    def generate(self,true_lbl,name):

        """Generate data from labels

        Args:
          true_lbl: true labels shaped (number of samples, size of label layer)
          name: string used for saving a video of generated visible activations
        """

        n_samples = true_lbl.shape[0]

        records = []
        fig,ax = plt.subplots(1,1,figsize=(3,3))
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        ax.set_xticks([]); ax.set_yticks([])

        lbl = true_lbl

        # [TODO TASK 4.2] fix the label in the label layer and run alternating Gibbs sampling in the top RBM. From the top RBM, drive the network \
        # top to the bottom visible layer (replace 'vis' from random to your generated visible layer).

        vis = np.random.randint(0,2,size=(n_samples,self.sizes["hid"]))

        rbm = self.rbm_stack['hid+lbl--top']

        for iter in range(self.n_gibbs_gener):
            vis = np.concatenate((vis,lbl),axis=1) #clamping labels
            #print(vis)
            #input()
            #print(vis[:,-10:])
            img_gen, vis = rbm.get_v_given_h(rbm.get_h_given_v(vis)[1])

            vis = vis[:,:-self.n_labels]
            img_gen = img_gen[:,:-self.n_labels]
            img_gen = self.rbm_stack['vis--hid'].get_v_given_h_dir(img_gen)[0] #one down
            records.append( [ ax.imshow(img_gen.reshape(self.image_size), cmap="bwr", vmin=0, vmax=1, animated=True, interpolation=None) ] )
        stitch_video(fig, records).save("videos/%s.generate%d.mp4"%(name, np.argmax(true_lbl)))

            # if iter == 0 or iter == 50 or iter == 100 or iter == 150:
            #     plt.imshow(img_gen.reshape(self.image_size))
            #     plt.show()


        return

    def train_greedylayerwise(self, vis_trainset, lbl_trainset, n_iterations):

        """
        Greedy layer-wise training by stacking RBMs. This method first tries to load previous saved parameters of the entire RBM stack.
        If not found, learns layer-by-layer (which needs to be completed) .
        Notice that once you stack more layers on top of a RBM, the weights are permanently untwined.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        try :

            self.loadfromfile_rbm(loc="trained_rbm",name="vis--hid")
            self.rbm_stack["vis--hid"].untwine_weights()

            self.loadfromfile_rbm(loc="trained_rbm",name="hid+lbl--top")

        except IOError :

            # [DONE TASK 4.2] use CD-1 to train all RBMs greedily

            print ("training vis--hid")
            """
            CD-1 training for vis--hid
            """
            rbm = self.rbm_stack['vis--hid']
            rbm.cd1(vis_trainset, n_iterations=n_iterations)
            self.savetofile_rbm(loc="trained_rbm",name="vis--hid")
            
            #top_trainset = rbm.get_h_given_v(vis_trainset)[0]
            #print ("training hid--pen")
            #self.rbm_stack["vis--hid"].untwine_weights()
            #"""
            #CD-1 training for hid--pen
            #"""
            #rbm = self.rbm_stack['hid--pen']
            #rbm.cd1(hid_trainset, n_iterations=n_iterations)
            #self.savetofile_rbm(loc="trained_rbm",name="hid--pen")
            #print(hid_trainset.shape, lbl_trainset.shape)
            
            top_trainset = np.concatenate((rbm.get_h_given_v(vis_trainset)[0], lbl_trainset), axis=1)
            print(top_trainset)

            print ("training hid+lbl--top")
            self.rbm_stack["vis--hid"].untwine_weights()
            """
            CD-1 training for hid+lbl--top
            """
            rbm = self.rbm_stack['hid+lbl--top']
            rbm.cd1(top_trainset, n_iterations=n_iterations)
            self.savetofile_rbm(loc="trained_rbm",name="hid+lbl--top")

        return

    def train_wakesleep_finetune(self, vis_trainset, lbl_trainset, n_iterations):

        """
        Wake-sleep method for learning all the parameters of network.
        First tries to load previous saved parameters of the entire network.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        print ("\ntraining wake-sleep..")
        batch_size = 20
        try :

            self.loadfromfile_dbn(loc="trained_dbn",name="vis--hid")
            self.loadfromfile_dbn(loc="trained_dbn",name="hid+lbl--top")

        except IOError :
            #rbmB, rbmH, rbmT = self.rbm_stack['vis--hid'], self.rbm_stack['hid--pen'], self.rbm_stack['pen+lbl--top']
            rbmH, rbmT = self.rbm_stack['vis--hid'], self.rbm_stack['hid+lbl--top']
            self.n_samples = vis_trainset.shape[0]
            self.n_labels = lbl_trainset.shape[1]
            mbs = 0

            for it in range(n_iterations):

                # [DONE TASK 4.3] wake-phase : drive the network bottom to top using fixing the visible and label data.
                y_0 = vis_trainset[mbs:(mbs+batch_size),:]
                lbl = lbl_trainset[mbs:(mbs+batch_size),:]
                mbs = (mbs + batch_size) % self.n_samples
                """Wake phase: memorize rec activations y_i"""
                y_1 = rbmH.get_h_given_v_dir(y_0)[1]
                #y_2 = rbmH.get_h_given_v_dir(y_1)[1]
                # [TODO TASK 4.3] alternating Gibbs sampling in the top RBM for k='n_gibbs_wakesleep' steps, also store neccessary information for learning this RBM.
                """Gibbs sampling in top"""
                v_0 = np.concatenate((y_1, lbl), axis=1)
                # first sampling : controlled to memorize h_0
                h_0, h_activation = rbmT.get_h_given_v(v_0)
                v_k, z_1 = rbmT.get_v_given_h(h_activation)
                for _ in range(self.n_gibbs_wakesleep - 1): # gibbs sampling in top
                    v_k, z_1 = rbmT.get_v_given_h(rbmT.get_h_given_v(z_1)[1])
                h_k = rbmT.get_h_given_v(v_k)[0]
                # [TODO TASK 4.3] sleep phase : from the activities in the top RBM, drive the network top to bottom.
                """Sleep phase: memorize gen activations z_i"""
                z_1 = z_1[:, :-self.n_labels]
                z_0 = rbmH.get_v_given_h_dir(z_1)[1]
                # [TODO TASK 4.3] compute predictions : compute generative predictions from wake-phase activations, and recognize predictions from sleep-phase activations.
                # Note that these predictions will not alter the network activations, we use them only to learn the directed connections.
                """Compute gen & rec predictions & update"""
                # [DONE TASK 4.3] update generative parameters : here you will only use 'update_generate_params' method from rbm class.

                y_P = rbmH.get_v_given_h_dir(y_1)[0]
                rbmH.update_generate_params(y_0, y_1, y_P)

                # [TODO TASK 4.3] update parameters of top rbm : here you will only use 'update_params' method from rbm class.
                rbmT.update_params(v_0, h_0, v_k, h_k)

                # [TODO TASK 4.3] update generative parameters : here you will only use 'update_recognize_params' method from rbm class.
                y_P = rbmH.get_h_given_v_dir(z_0)[0]
                rbmH.update_recognize_params(z_1, z_0, y_P)

                if it % self.print_period == 0 : print ("iteration=%7d"%it)
                sys.stdout.flush()

            ### no save till the method doesn't work (:
            self.savetofile_dbn(loc="trained_dbn",name="vis--hid")
            self.savetofile_dbn(loc="trained_dbn",name="hid+lbl--top")

        return


    def loadfromfile_rbm(self,loc,name):

        self.rbm_stack[name].weight_vh = np.load("%s/rbm.%s.weight_vh.npy"%(loc,name))
        self.rbm_stack[name].bias_v    = np.load("%s/rbm.%s.bias_v.npy"%(loc,name))
        self.rbm_stack[name].bias_h    = np.load("%s/rbm.%s.bias_h.npy"%(loc,name))
        print ("loaded rbm[%s] from %s"%(name,loc))
        return

    def savetofile_rbm(self,loc,name):

        np.save("%s/rbm.%s.weight_vh"%(loc,name), self.rbm_stack[name].weight_vh)
        np.save("%s/rbm.%s.bias_v"%(loc,name),    self.rbm_stack[name].bias_v)
        np.save("%s/rbm.%s.bias_h"%(loc,name),    self.rbm_stack[name].bias_h)
        return

    def loadfromfile_dbn(self,loc,name):

        self.rbm_stack[name].weight_v_to_h = np.load("%s/dbn.%s.weight_v_to_h.npy"%(loc,name),allow_pickle=True)
        self.rbm_stack[name].weight_h_to_v = np.load("%s/dbn.%s.weight_h_to_v.npy"%(loc,name),allow_pickle=True)
        self.rbm_stack[name].bias_v        = np.load("%s/dbn.%s.bias_v.npy"%(loc,name),allow_pickle=True)
        self.rbm_stack[name].bias_h        = np.load("%s/dbn.%s.bias_h.npy"%(loc,name),allow_pickle=True)
        print ("loaded rbm[%s] from %s"%(name,loc))
        return

    def savetofile_dbn(self,loc,name):

        np.save("%s/dbn.%s.weight_v_to_h"%(loc,name), self.rbm_stack[name].weight_v_to_h)
        np.save("%s/dbn.%s.weight_h_to_v"%(loc,name), self.rbm_stack[name].weight_h_to_v)
        np.save("%s/dbn.%s.bias_v"%(loc,name),        self.rbm_stack[name].bias_v)
        np.save("%s/dbn.%s.bias_h"%(loc,name),        self.rbm_stack[name].bias_h)
        return
