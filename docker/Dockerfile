FROM ubuntu:18.04

#RUN useradd --create-home astuser
#WORKDIR /home/astuser


COPY docker/docker_setup.sh ./
#RUN ["chmod", "+x", "./docker_setup.sh"]
RUN ./docker_setup.sh
#RUN apt-get -y update && apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa
#RUN apt-get update && apt-get install -y python3.6 python3.6-dev build-essential
#RUN apt install -y libdb-dev python3-bsddb3
#RUN python3 -m pip install --upgrade pip
#RUN apt install -y python3-pip
#RUN pip3 install --upgrade pip
#RUN apt-get update

# Create a user so there is no root access and switch to its home dir
RUN useradd --create-home astuser
WORKDIR /home/astuser

# Copy needed files and give astuser access
COPY requirements.txt ./
COPY . ./AdaptiveStressTestingToolbox
RUN chown astuser ./requirements.txt
RUN chown -R astuser ./AdaptiveStressTestingToolbox

#Switch to astuser
USER astuser

ENV VIRTUAL_ENV=/home/astuser/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"


#RUN python3.6 -m venv ./venv
#RUN . ./venv/bin/activate
RUN pip install --upgrade pip
RUN pip install --ignore-installed -r ./requirements.txt

RUN pip install --ignore-installed -e ./AdaptiveStressTestingToolbox
#RUN apt install -y python-pip && pip --version
#RUN pip install --upgrade pip

ENV PYTHONPATH "/tmp/AdaptiveStressTestingToolbox/src:/tmp/AdaptiveStressTestingToolbox/examples:/tmp/AdaptiveStressTestingToolbox/tests:/tmp/AdaptiveStressTestingToolbox/third_party/garage/src"
ENV PYTHONHASHSEED 0
#ENTRYPOINT . ./venv/bin/activate

COPY docker/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
