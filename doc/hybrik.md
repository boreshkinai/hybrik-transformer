
Note that the latest HybrIK version as of 15.12.2021, commit ae1bc3cea0cc5aa98fb512eeb295c3478b0c598f, is incompatible with the pytorch versions 1.9.0 nor 1.2.0 (claimed to be default in https://github.com/Jeff-sjtu/HybrIK). You will have a stack trace, which can be fixed by replacing lines 245-250 with the following code:
```
hm_x = hm_x * torch.arange(hm_x.shape[-1]).to(hm_x)
hm_y = hm_y * torch.arange(hm_y.shape[-1]).to(hm_y)
hm_z = hm_z * torch.arange(hm_z.shape[-1]).to(hm_z)
```


