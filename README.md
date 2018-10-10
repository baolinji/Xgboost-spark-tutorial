Xgboost使用与分析
===
1.源码编译安装
---  
          在spark上如何编译使用xgboost，尽量不要在windows编译，因为cmake会产生不同的文件，在linux上使用会  
      提示缺少/lib/libxgboost.so。主要步骤如下(具体解释可参照官方文档)：
      
         1.从git 拉取源码(因为xgboost用到了自带的rabbit,拉取代码需要加recursive，初始化所有模块)：
```
    git clone --recursive https://github.com/dmlc/xgboost
    cd xgboost; make -j4
    note:需要注意make版本,在3.3以上。旧版本需要更新下，或者临时替换make版本。
```
      
         2.进入jvm-package子目录，使用mvn进行jar包编译:
```
     mvn install
     or mvn -DskipTests install
```
          在xgboost4j,xgboost4j-spark,xgboost-flink三个子目录下的target目录生成jar包，如下图所示。
    
          由于在linux平台编译生成的jar包不能在其他系统上使用，windows产生的是xgboost4j.dll文件。使用xgboost4j-spark:  
      在pom.xml添加依赖，由于产生的jar不带依赖，所以要是在spark上使用需要加xgboost4j的依赖,如下所示。
```
    <dependency>
        <groupId>ml.dmlc</groupId>
        <artifactId>xgboost4j-spark</artifactId>
        <version>0.81</version>
    </dependency>

    <dependency>
        <groupId>ml.dmlc</groupId>
        <artifactId>xgboost4j</artifactId>
        <version>0.81</version>
    </dependency>
```
