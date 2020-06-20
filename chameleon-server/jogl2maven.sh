#!/usr/bin/env bash

#filename specified
if [ $# != 3 ]; then
        echo "usage: $0 distroUrl maven_repo version"
        exit
else
        echo "Processing: $1"
fi


BASEURL=$1
MVN_REPO=$2
VERSION=$3

TEMPFILE=./download/jogamp/$VERSION
mkdir -p "$TEMPFILE"
echo "using temp directory $TEMPFILE"

# base jogl class jar
echo "installing core jogl deps"
wget -nc $BASEURL/jar/jogl-all.jar -O "$TEMPFILE/jogl-all.jar"
mvn install:install-file -Dversion=$VERSION -DlocalRepositoryPath="$MVN_REPO" -DgroupId=org.jogamp.jogl -DartifactId=jogl-all -Dpackaging=jar -Dfile="$TEMPFILE/jogl-all.jar" -DgeneratePom=true

echo "installing gluegen"
wget -nc $BASEURL/jar/gluegen-rt.jar -O "$TEMPFILE/gluegen-rt.jar"
mvn install:install-file -Dversion=$VERSION  -DlocalRepositoryPath="$MVN_REPO" -DgroupId=org.jogamp.gluegen -DartifactId=gluegen-rt -Dpackaging=jar -Dfile="$TEMPFILE/gluegen-rt.jar" -DgeneratePom=true

# native jars
echo "installing native libs"
NATIVE_PLATFORMS=" linux-amd64 linux-i586 linux-armv6hf linux-armv6  linux-aarch64 macosx-universal windows-amd64 windows-i586 android-armv6 android-aarch64 android-x86 ios-amd64 ios-arm64"
for platform in $NATIVE_PLATFORMS; do
	echo " -> installing native lib for $platform"
	
	wget -nc $BASEURL/jar/jogl-all-natives-$platform.jar -O "$TEMPFILE/jogl-all-natives-$platform.jar"
	mvn install:install-file -DlocalRepositoryPath="$MVN_REPO" -Dclassifier="natives-$platform" -Dversion=$VERSION -DgroupId=org.jogamp.jogl -DartifactId=jogl-all -Dpackaging=jar -Dfile="$TEMPFILE/jogl-all-natives-$platform.jar" -DgeneratePom=true
	
	wget -nc $BASEURL/jar/gluegen-rt-natives-$platform.jar -O "$TEMPFILE/gluegen-rt-natives-$platform.jar"
	mvn install:install-file -DlocalRepositoryPath="$MVN_REPO" -Dclassifier="natives-$platform" -Dversion=$VERSION -DgroupId=org.jogamp.gluegen -DartifactId=gluegen-rt -Dpackaging=jar -Dfile="$TEMPFILE/gluegen-rt-natives-$platform.jar" -DgeneratePom=true
done

