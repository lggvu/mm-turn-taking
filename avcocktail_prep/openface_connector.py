import docker
import os
import cv2
import tarfile
from io import StringIO

'''
    Class to run containerised version of OpenFace
'''


class OpenFaceConnector:

    def __init__(self):

        # initialise the container and detach it
        self._client = docker.from_env()
        self._container = self._client.containers.run("algebr/openface:latest", detach=True, tty=True)

        self._homedir = '/home/openface-build'

    def start_container(self):
        pass

    def close_container(self):
        self._container.stop()

    # to copy to the container via tarfile
    def copy_to(self, src, dst):

        tmptar = src + '.tar'

        tar = tarfile.open(src+'.tar', 'w')
        tar.add(src, os.path.join('data', os.path.basename(src)))
        tar.close()

        data = open(tmptar, 'rb').read()
        self._container.put_archive(dst, data)

        # remove the tarfile
        os.remove(tmptar)

    # to copy from a directory from the container to machine
    def copy_from(self, src, dst):

        # a tmp place
        tar_tmp = os.path.join(dst, 'processed.tar')

        bits, status = self._container.get_archive(src)
        with open(tar_tmp, 'wb') as f:
            for chunk in bits:
                f.write(chunk)

        # extract the .tar
        tar = tarfile.open(tar_tmp)
        tar.extractall(dst)
        tar.close()

        # remove the tmp tar
        os.remove(tar_tmp)

    # execute a command on the running container
    def execute(self, cmd, stream_output=True, detach=True):
        _, stream = self._container.exec_run(cmd, stream=stream_output, detach=detach)
        res = ' '.join(s.decode() for s in stream)
        if stream:
            print('>> ', cmd, '\n', res)
        return res

    # process video from a path
    def process_video(self, video_path):

        # copy the video to container
        self.copy_to(video_path, self._homedir)

        # run the command on the video
        video_file = os.path.basename(video_path)

        # run the command for getting the pose,
        # single person in a single video
        self.execute(f'build/bin/FeatureExtraction -f "data/{video_file}"', stream_output=True, detach=False)

    def fetch_results(self, dst_dir):

        # copy back the processed dir
        self.copy_from(os.path.join(self._homedir, 'processed'), dst_dir)
