import numpy as np
import os

def save_wp(savedir, red_results, blue_results, all_results, magbins):
    
     # Save the results to text files in the format we want, and also save the covariance matrix as numpy array
    for i in range(len(red_results)):
        red_wp, red_cov = red_results[i]
        blue_wp, blue_cov = blue_results[i]
        all_wp, all_cov = all_results[i]

        # Currently we're choosing not to use the full covariance matrix, just the diagonal for our chi squared
        # since the result of the jackknife tests was kinda weird correlation matrices.

        # Format is: rp wp wp_err
        if red_wp is not None:
            with open(os.path.join(savedir, f'wp_red_M{-magbins[i]:d}.dat'), 'w') as f:
                for j in range(len(red_wp)):
                    f.write(f'{red_wp[j,0]:.8f} {red_wp[j,2]:.8f} {red_wp[j,3]:.8f}\n')
            np.save(os.path.join(savedir, f'wp_red_M{-magbins[i]:d}_cov.npy'), red_cov)

        if blue_wp is not None:
            with open(os.path.join(savedir, f'wp_blue_M{-magbins[i]:d}.dat'), 'w') as f:
                for j in range(len(blue_wp)):
                    f.write(f'{blue_wp[j,0]:.8f} {blue_wp[j,2]:.8f} {blue_wp[j,3]:.8f}\n')
            np.save(os.path.join(savedir, f'wp_blue_M{-magbins[i]:d}_cov.npy'), blue_cov)
            
        if all_wp is not None:
            with open(os.path.join(savedir, f'wp_all_M{-magbins[i]:d}.dat'), 'w') as f:
                for j in range(len(all_wp)):
                    f.write(f'{all_wp[j,0]:.8f} {all_wp[j,2]:.8f} {all_wp[j,3]:.8f}\n')
            np.save(os.path.join(savedir, f'wp_all_M{-magbins[i]:d}_cov.npy'), all_cov)