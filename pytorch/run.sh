# For training
python main.py --dataset jenny --dataroot data/jenny/ --outf models --manualSeed 9999 --cuda --batchSize 128 --mode train

# For testing 
python main.py --dataset jenny --dataroot data/jenny --outf models --manualSeed 9999 --cuda --batchSize 128 --mode test --testdata_dir tts_test_data --netG models/netG_epoch_24.pth

