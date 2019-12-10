from osgeo import gdal, gdal_array
import numpy as np

def GetSuffix(filepath):
    i = filepath.rfind('.')
    str = ""
    if i == -1 or i == 0 or (i == 1 and filepath[0] == '.') or filepath[i + 1].isdigit() or len(filepath) - i > 5:
        return str
    else:
        str = filepath[i + 1:len(filepath)]
        return str


def GetFilePath(filepath):
    i = filepath.rfind('.')
    str = ""
    if i == -1 or i == 0 or (i == 1 and filepath[0] == '.') or filepath[i + 1].isdigit():
        return filepath
    else:
        str = filepath[0:i]
        return str


class CXImage():
    '''
        m_nBands：波段数
        m_nLines：height
        m_nSamples：width
        m_nDataType：数据类型
        m_strImgPath：图像路径
        currentHeight：当前窗口高度
        currentWidth：当前窗口宽度
        partWidth：分块宽度
        partHeight：分块高度
        proDataset：保存图像信息
        currentPosX：当前X坐标
        currentPosY：当前Y坐标
        data_arrange == 0 <==> (height, width, band)
                     == 1 <==> (band, height, width)(gdal)
    '''
    m_nBands = 0
    m_nLines = 0
    m_nSamples = 0
    m_nClasses = 0

    m_nDataType = np.uint8
    m_strImgPath = ""

    currentHeight = 0
    currentWidth = 0
    partWidth = 0
    partHeight = 0
    isNormalization = True

    proDataset = None
    currentPosX = 0
    currentPosY = 0

    minBandValue = None
    maxBandValue = None

    # data_arrange = 0

    # 创建对象时的初始化操作
    def __init__(self, nBands=0, nLines=0, nSamples=0, nDataType=np.uint8, nClass=0, strImgPath=None):
        self.m_nBands = nBands
        self.m_nLines = nLines
        self.m_nSamples = nSamples
        self.m_nDataType = nDataType
        self.m_nClasses = nClass
        self.m_strImgPath = strImgPath
        # self.data_arrange = data_arrange

        self.minBandValue = None
        self.maxBandValue = None
        self.isNormalization = False
        self.proDataset = None

    # 注册图像驱动，同时设置输出图像数据类型
    def Create(self, nBands, nLines, nSamples, nDataType, strImgPath, nClass=0):
        '''
            np.uint8 <==> gdal.GDT_Byte == 1
            np.int8 <==> gdal.GDT_Byte == 1
            np.byte <==> gdal.GDT_Byte == 1
            np.uint16 <==> gdal.GDT_UInt16 == 2
            np.int16 <==> gdal.GDT_Int16 == 3
            np.uint <==> gdal.GDT_UInt32 == 4
            np.uint32 <==> gdal.GDT_UInt32 == 4
            np.int32 <==> gdal.GDT_Int32 == 5
            np.float32 <==> gdal.GDT_Float32 == 6
            np.float64 <==> gdal.GDT_Float64 == 7
            np.complex64 <==> gdal.GDT_CFloat32 == 10
            np.complex128 <==> gdal.GDT_CFloat64 == 11
            np.float <==> None
            np.int <==> None
            np.complex <==> None
        '''
        nDataType_result = gdal_array.NumericTypeCodeToGDALTypeCode(nDataType)

        self.m_nBands = nBands
        self.m_nLines = nLines
        self.m_nSamples = nSamples
        self.m_nDataType = nDataType
        self.m_strImgPath = strImgPath
        self.m_nClasses = nClass

        self.partWidth = self.m_nSamples
        self.partHeight = self.m_nLines
        self.currentPosX = 0
        self.currentPosY = 0
        # self.currentHeight = self.m_nLines
        # self.currentWidth = self.m_nSamples

        self.initNextWidowsSize()

        suffix = GetSuffix(self.m_strImgPath)
        if suffix == "bmp":
            driver = gdal.GetDriverByName("BMP")
        elif suffix == "jpg":
            driver = gdal.GetDriverByName("JPEG")
        elif suffix == "tif" or suffix == "tiff":
            driver = gdal.GetDriverByName("GTiff")
        elif suffix == "bt":
            driver = gdal.GetDriverByName("BT")
        elif suffix == "ecw":
            driver = gdal.GetDriverByName("ECW")
        elif suffix == "fits":
            driver = gdal.GetDriverByName("FITS")
        elif suffix == "gif":
            driver = gdal.GetDriverByName("GIF")
        elif suffix == "hdf":
            driver = gdal.GetDriverByName("HDF4")
        elif suffix == "hdr":
            driver = gdal.GetDriverByName("EHdr")
        elif suffix == "" or suffix == "img":
            driver = gdal.GetDriverByName("ENVI")
        else:
            print("GetDriverByName Error!")
            return

        # 保存图像到m_strImgPath路径，宽度为m_nSamples，高度为m_nLines，波段数为m_nBands，读入的数据为GDALDataType格式
        self.proDataset = driver.Create(self.m_strImgPath, nSamples, nLines, self.m_nBands,
                                        nDataType_result)

    # 打开文件，保存信息
    def Open(self, strImgPath):
        self.m_strImgPath = strImgPath

        self.proDataset = gdal.Open(self.m_strImgPath)
        if self.proDataset == None:
            print("file open error")
            return False
        self.m_nBands = self.proDataset.RasterCount
        self.m_nLines = self.proDataset.RasterYSize
        self.m_nSamples = self.proDataset.RasterXSize

        self.partWidth = self.m_nSamples
        self.partHeight = self.m_nLines
        self.currentPosX = 0
        self.currentPosY = 0
        self.currentHeight = self.m_nLines
        self.currentWidth = self.m_nSamples

        self.initNextWidowsSize()
        return True

    # 跳到下一个窗口
    def next(self, padding=10):
        self.currentPosX += self.currentWidth - padding
        self.currentPosY += 0

        if self.currentPosX + padding >= self.m_nSamples:
            self.currentPosX = 0
            self.currentPosY += self.currentHeight - padding
        if self.currentPosY + padding >= self.m_nLines:
            self.currentPosX = 0
            self.currentPosY = 0
            return False

        self.initNextWidowsSize()
        return True

    # 设置分块大小
    def setPartSize(self, setPartWidth=-1, setPartHeight=-1, memorySize=-1, typeSize=4, safetyFactor=6):
        # 设置了setPartWidth和setPartHeight参数
        if setPartWidth != -1 and setPartHeight != -1:
            self.partHeight = setPartHeight
            self.partWidth = setPartWidth
            self.currentPosX = 0
            self.currentPosY = 0
            self.initNextWidowsSize()
            return
        # 只设置了setPartWidth参数
        elif setPartWidth != -1 and setPartHeight == -1:
            ratio = float(self.m_nLines) / self.m_nSamples
            self.partWidth = setPartWidth
            self.partHeight = int(self.partWidth * ratio)
            self.currentPosX = 0
            self.currentPosY = 0
            self.initNextWidowsSize()
            return
        # 只设置了memorySize参数，根据设置内存大小分块
        elif setPartWidth == -1 and setPartHeight == -1 and memorySize != -1:
            if memorySize < 1000:
                self.partWidth = 0
                self.partHeight = 0
                return
            ratio = float(self.m_nLines) / self.m_nSamples
            self.partWidth = int((memorySize * 1000.0 / (typeSize * ratio * safetyFactor)) ** 0.5)
            self.partHeight = int(self.partWidth * ratio)
            self.currentPosX = 0
            self.currentPosY = 0
            self.initNextWidowsSize()
        # 无参数
        elif setPartWidth == -1 and setPartHeight == -1 and memorySize == -1:
            self.partHeight = self.m_nLines
            self.partWidth = self.m_nSamples
            self.currentPosX = 0
            self.currentPosY = 0
            self.initNextWidowsSize()
        else:
            print("setPartSize parameters error!")
            return

    # 设置类别编号
    def setClassNo(self, classNo):
        self.m_nClasses = classNo
        if self.m_nBands == 1 and self.m_nClasses > 0:
            # 按默认方式设置分类颜色
            self.setClassificationColor()
        return

    # 根据ROI文件或以默认方式设置分类颜色 abuilding...
    def setClassificationColor(self, trainFile=""):
        isUsedDefault = False
        if trainFile == "":
            isUsedDefault = True
        Is = open(trainFile, "r")
        if Is == None:
            print("Open ROI file Error!")
            isUsedDefault = True
        pIn = ""
        if not isUsedDefault:
            pIn = Is.readline()
            pIn = Is.readline()
            self.m_nClasses = pIn.split()[4]
            pIn = Is.readline()

        usedColor = np.zeros(3 * (int(self.m_nClasses) + 1), dtype=int)
        pColor = [0, 0, 0,   0, 0, 255,   46, 139, 87,   0, 255, 0,   216, 191, 216,   255, 0, 0,   255, 255, 255,
                   255, 255, 0,   0, 255, 255,   255, 0, 255,   48, 48, 48,   128, 0, 0,   0, 128, 0,   0, 0, 128,
                  128, 128, 0,   0, 128, 128,   128, 0, 128,   255, 128, 0,   128, 255, 0,   255, 0, 128]

        colorTable = gdal.ColorTable(gdal.GPI_RGB)
        pBand = self.proDataset.GetRasterBand(1)

        for i in range(int(self.m_nClasses) + 1):
            if isUsedDefault or i == 0:
                usedColor[3 * i] = pColor[(3 * i) % 60]
                usedColor[3 * i + 1] = pColor[(3 * i + 1) % 60]
                usedColor[3 * i + 2] = pColor[(3 * i + 2) % 60]
            else:
                pIn = Is.readline()  # 读空行
                pIn = Is.readline()  # ; ROI name: Random Sample (salinas_gt_byte / Class #1)
                pIn = Is.readline()  # ; ROI rgb value: {255, 0, 0}
                strTemp = pIn.split()[4]  # {255,
                str = strTemp[1:len(strTemp) - 1]  # 读255
                usedColor[3 * i] = int(str)

                strTemp = pIn.split()[5]  # 0,
                str = strTemp[0:1]
                usedColor[3 * i + 1] = int(str)

                strTemp = pIn.split()[6]  # 0}
                str = strTemp[0:1]
                usedColor[3 * i + 2] = int(str)

                pIn = Is.readline()  # ; ROI npts: 201
            colorEntry = gdal.ColorEntry()
            colorEntry.c1 = usedColor[3 * i]
            colorEntry.c2 = usedColor[3 * i + 1]
            colorEntry.c3 = usedColor[3 * i + 2]
            colorEntry.c4 = 0
            colorTable.SetColorEntry(i, colorEntry)

        return pBand.SetColorTable(colorTable)

    # 获取分块数量
    def getPartionNum(self, padding=10):
        temp_currentPosX = self.currentPosX
        temp_currentPosY = self.currentPosY
        temp_currentHeight = self.currentHeight
        temp_currentWidth = self.currentWidth

        self.currentPosX = 0
        self.currentPosY = 0
        self.initNextWidowsSize()
        partionNum = 1
        maxPartionWidth = 0
        maxPartionHeight = 0

        while True:
            if maxPartionHeight < self.currentHeight:
                maxPartionHeight = self.currentHeight
            if maxPartionWidth < self.currentWidth:
                maxPartionWidth = self.currentWidth
            if not self.next(padding):
                break
            partionNum += 1
        self.currentPosX = temp_currentPosX
        self.currentPosY = temp_currentPosY
        self.currentHeight = temp_currentHeight
        self.currentWidth = temp_currentWidth
        return partionNum, maxPartionHeight, maxPartionWidth

    # 调整分块参数
    def setPartionParameters(self, currentPosXValue=-1, currentPosYValue=-1, currentWidthValue=-1,
                             currentHeightValue=-1):
        if currentHeightValue > 0 and currentPosYValue > -1 and currentPosXValue > -1 and currentWidthValue > 0:
            self.currentHeight = currentHeightValue
            self.currentWidth = currentWidthValue
            self.currentPosX = currentPosXValue
            self.currentPosY = currentPosYValue

    def initMinMaxBandValue(self, is2PercentScale=False, percentValue=0.02):
        return

    def uninitMinMaxBandValue(self):
        return

    def setNormalizaion(self, setIsNormalization=True):
        self.isNormalization = setIsNormalization
        return

    # 设置头信息
    def setHeaderInformation(self, img):
        # 获取投影信息
        projectionRef = img.proDataset.GetProjectionRef()
        # 写入投影
        self.proDataset.SetProjection(projectionRef)

        # 获取仿射矩阵信息
        padfTransform = img.proDataset.GetGeoTransform()
        # 写入仿射变换参数
        self.proDataset.SetGeoTransform(padfTransform)

        # GCPCount = img.proDataset.GetGCPCount()
        # GCPProjection = img.proDataset.GetGCPProjection()
        # getGCPS = img.proDataset.GetGCPs()
        # self.proDataset.SetGCPs(GCPCount, getGCPS, GCPProjection)

    # 获取图像栅格数据，转化为dataType类型数组并返回
    def GetData(self, dataType, cHeight=-1, cWidth=-1, cPosX=-1, cPosY=-1, data_arrange=0):
        '''
            data_arrange == 0 <==> (height, width, band)
                         == 1 <==> (band, height, width)(gdal)
        '''
        # 没有设置cHeight、cWidth、cPosX、cPosY，即根据当前类内参数读取
        if cHeight == -1 and cWidth == -1 and cPosX == -1 and cPosY == -1:

            data = self.proDataset.ReadAsArray(self.currentPosX, self.currentPosY, self.currentWidth,
                                               self.currentHeight)
        # 根据设定参数读取
        elif cHeight != -1 and cWidth != -1 and cPosX != -1 and cPosY != -1:
            data = self.proDataset.ReadAsArray(cPosX, cPosY, cWidth, cHeight)
        else:
            raise AttributeError("GetData parameters error!")

        if data.ndim == 3 and data_arrange == 0:
            data = data.transpose((1, 2, 0))
        elif data.ndim == 3 and data_arrange == 1:
            pass
        elif data.ndim == 2:
            pass
        else:
            raise AttributeError("data shape error!")

        data = data.astype(dataType)

        return data

    def NormlizedData(self, data, setMin=0, setMax=1):
        return

    # 根据pData数据，并转化为Create时的类型对图像进行写入操作
    def WriteImgData(self, pData, cHeight=-1, cWidth=-1, cPosX=-1, cPosY=-1, padding=10, data_arrange=0):
        '''
            data_arrange == 0 <==> (height, width, band)
                         == 1 <==> (band, height, width)(gdal)
        '''
        # 直接写整幅图像
        if cHeight == self.m_nLines and cWidth == self.m_nSamples and cPosX == 0 and cPosY == 0:
            temp_currentPosX = self.currentPosX
            temp_currentPosY = self.currentPosY
            temp_currentWidth = self.currentWidth
            temp_currentHeight = self.currentHeight
            if data_arrange == 0 and pData.ndim == 3:
                temp_data = pData.transpose((2, 0, 1))
            else:
                temp_data = pData
        # 无参数图像写入
        elif cHeight == -1 and cWidth == -1 and cPosX == -1 and cPosY == -1:
            temp_currentHeight, temp_currentWidth, temp_currentPosX, temp_currentPosY, temp_data = self.ProcessDataBeforeWrite(
                pData, self.currentHeight, self.currentWidth, self.currentPosX, self.currentPosY, padding, data_arrange)
        # 有参数图像写入
        elif cHeight != -1 and cWidth != -1 and cPosX != -1 and cPosY != -1:
            temp_currentHeight, temp_currentWidth, temp_currentPosX, temp_currentPosY, temp_data = self.ProcessDataBeforeWrite(
                pData, cHeight, cWidth, cPosX, cPosY, padding, data_arrange)
        else:
            raise AttributeError("WriteImgData parameters error!")

        temp_data = temp_data.astype(self.m_nDataType)
        self.proDataset.WriteRaster(int(temp_currentPosX), int(temp_currentPosY), int(temp_currentWidth),
                                        int(temp_currentHeight), temp_data.tobytes())
        self.proDataset.FlushCache()

        if not np.all(temp_data == 0):
            del temp_data
        return True

    # 对图像数据进行处理以正确写入
    def ProcessDataBeforeWrite(self, pData, cHeight, cWidth, cPosX, cPosY, padding, data_arrange):
        '''
            data_arrange == 0 <==> (height, width, band)
                         == 1 <==> (band, height, width)(gdal)
        '''
        temp_currentPosX = cPosX
        temp_currentPosY = cPosY
        temp_currentWidth = cWidth
        temp_currentHeight = cHeight
        if cPosX > 0:
            temp_currentPosX = cPosX + padding / 2
            temp_currentWidth = cWidth - padding / 2
        if cPosY > 0:
            temp_currentPosY = cPosY + padding / 2
            temp_currentHeight = cHeight - padding / 2
        if cPosX + cWidth < self.m_nSamples:
            temp_currentWidth = temp_currentWidth - padding / 2
        if cPosY + cHeight < self.m_nLines:
            temp_currentHeight = temp_currentHeight - padding / 2

        if data_arrange == 0 and pData.ndim == 3:
            pData = pData.transpose((2, 0, 1))

        if pData.ndim == 3:
            if cPosX == 0 and cPosY == 0:
                temp_data = pData[:, :int(temp_currentHeight), :int(temp_currentWidth)]
            elif cPosX > 0 and cPosY == 0:
                temp_data = pData[:, :int(temp_currentHeight),
                            int(padding / 2):int(temp_currentWidth + padding / 2)]
            elif cPosX > 0 and cPosY > 0:
                temp_data = pData[:, int(padding / 2):int(temp_currentHeight + padding / 2),
                            int(padding / 2):int(temp_currentWidth + padding / 2)]
            else:
                temp_data = pData[:, int(padding / 2):int(temp_currentHeight + padding / 2),
                            :int(temp_currentWidth)]
        elif pData.ndim == 2:
            if cPosX == 0 and cPosY == 0:
                temp_data = pData[:int(temp_currentHeight), :int(temp_currentWidth)]
            elif cPosX > 0 and cPosY == 0:
                temp_data = pData[:int(temp_currentHeight),
                            int(padding / 2):int(temp_currentWidth + padding / 2)]
            elif cPosX > 0 and cPosY > 0:
                temp_data = pData[int(padding / 2):int(temp_currentHeight + padding / 2),
                            int(padding / 2):int(temp_currentWidth + padding / 2)]
            else:
                temp_data = pData[int(padding / 2):int(temp_currentHeight + padding / 2),
                            :int(temp_currentWidth)]
        else:
            raise AttributeError("pData.ndim error!")
        return temp_currentHeight, temp_currentWidth, temp_currentPosX, temp_currentPosY, temp_data

    # 初始化窗口大小，以及设置下一个窗口大小
    def initNextWidowsSize(self):
        acceptWidth = 0.5 * self.partWidth
        acceptHeight = 0.5 * self.partHeight

        self.currentWidth = self.partWidth
        self.currentHeight = self.partHeight

        if self.m_nSamples <= self.currentWidth:
            self.currentWidth = self.m_nSamples
        if self.m_nLines <= self.currentHeight:
            self.currentHeight = self.m_nLines

        if self.currentPosX + self.partWidth > self.m_nSamples or acceptWidth + self.currentPosX + self.partWidth > self.m_nSamples:
            self.currentWidth = self.m_nSamples - self.currentPosX
        if self.currentPosY + self.partHeight > self.m_nLines or acceptHeight + self.currentPosY + self.partHeight > self.m_nLines:
            self.currentHeight = self.m_nLines - self.currentPosY