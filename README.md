# carbon-footprint
A repository for `codecarbon` logs.

## How to Track an Experiment
We use `codecarbon` to track the carbon footprint for experiments.

If you are using `huggingface/transforers`, `codecarbon` has already been integrated. You only need to upgrade to the latest version and add `--report_to codecarbon` and the carbon footprint log `emission.csv` will be automatically saved with the model.

If you are not using `huggingface/transformers`, you should integrate `codecarbon` to your code. Please refer to the instruction [here](https://github.com/mlco2/codecarbon) for how to integrate (it's super easy!) and an example of integration can be found [here](https://github.com/huggingface/transformers/pull/12304/files).

## How to Upload the Logs
1. Create a folder named with your WG in the root of this repo (e.g., `carbon-footprint-wg`). [Example](https://github.com/bigscience-workshop/carbon-footprint/tree/master/carbon-footprint-wg)
2. Create subfolders (e.g., `prompt-enginerring`, `architecture-proof-of-concept`).
3. Run the experiments and find your `emission.csv` file saved together with the model.
4. Upload the logs into the subfolders. Make sure they are `.csv` file. You can rename `emission.csv` to any name you like.
5. Done! Have a coffee! ☕

## Useful Commands
### Rename `emission.csv` to a unique file name with its timestamp
```bash
export f=emission.csv
mv -n "$f" "emission_$(date -r "$f" +"%Y%m%d_%H%M%S").csv"
```

## FAQ

### Do I need code review to upload the logs?

No. Just upload the logs with the Github UI or Git and open a pull request. These uploads will be merged periodically. Also you can ask for write permission if you don't have it yet.
