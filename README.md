# fastai_api
a simple restful api to test a trained model with fastai

For this to work, you have to get the fastai librairy https://github.com/fastai/fastai/ and follow the instructions.

Then you have to create a symlink (mklink /D your_path_to_this_project/fastai_api/server/fastai your_path_to_fastai_librairy/fastai/fastai) or copy the fastai librairy in the server folder


To run the server : cd fastai_api/server
                    python run_fastai_server.py
                    
To request the server : cd fastai_api/client
                    python request.py
