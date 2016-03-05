-- DROP TABLE udacity.pisa2012;

CREATE DATABASE udacity;

CREATE TABLE udacity.pisa2012 (input_id BIGINT(6), CNT DOUBLE, SUBNATIO DOUBLE, STRATUM DOUBLE, OECD DOUBLE, NC DOUBLE, SCHOOLID DOUBLE, STIDSTD DOUBLE, ST01Q01 DOUBLE, ST02Q01 DOUBLE, ST03Q01 DOUBLE, ST03Q02 DOUBLE, ST04Q01 DOUBLE, ST05Q01 DOUBLE, ST06Q01 DOUBLE, ST07Q01 DOUBLE, ST07Q02 DOUBLE, ST07Q03 DOUBLE, ST08Q01 DOUBLE, ST09Q01 DOUBLE, ST115Q01 DOUBLE, ST11Q01 DOUBLE, ST11Q02 DOUBLE, ST11Q03 DOUBLE, ST11Q04 DOUBLE, ST11Q05 DOUBLE, ST11Q06 DOUBLE, ST13Q01 DOUBLE, ST14Q01 DOUBLE, ST14Q02 DOUBLE, ST14Q03 DOUBLE, ST14Q04 DOUBLE, ST15Q01 DOUBLE, ST17Q01 DOUBLE, ST18Q01 DOUBLE, ST18Q02 DOUBLE, ST18Q03 DOUBLE, ST18Q04 DOUBLE, ST19Q01 DOUBLE, ST20Q01 DOUBLE, ST20Q02 DOUBLE, ST20Q03 DOUBLE, ST21Q01 DOUBLE, ST25Q01 DOUBLE, ST26Q01 DOUBLE, ST26Q02 DOUBLE, ST26Q03 DOUBLE, ST26Q04 DOUBLE, ST26Q05 DOUBLE, ST26Q06 DOUBLE, ST26Q07 DOUBLE, ST26Q08 DOUBLE, ST26Q09 DOUBLE, ST26Q10 DOUBLE, ST26Q11 DOUBLE, ST26Q12 DOUBLE, ST26Q13 DOUBLE, ST26Q14 DOUBLE, ST26Q15 DOUBLE, ST26Q16 DOUBLE, ST26Q17 DOUBLE, ST27Q01 DOUBLE, ST27Q02 DOUBLE, ST27Q03 DOUBLE, ST27Q04 DOUBLE, ST27Q05 DOUBLE, ST28Q01 DOUBLE, ST29Q01 DOUBLE, ST29Q02 DOUBLE, ST29Q03 DOUBLE, ST29Q04 DOUBLE, ST29Q05 DOUBLE, ST29Q06 DOUBLE, ST29Q07 DOUBLE, ST29Q08 DOUBLE, ST35Q01 DOUBLE, ST35Q02 DOUBLE, ST35Q03 DOUBLE, ST35Q04 DOUBLE, ST35Q05 DOUBLE, ST35Q06 DOUBLE, ST37Q01 DOUBLE, ST37Q02 DOUBLE, ST37Q03 DOUBLE, ST37Q04 DOUBLE, ST37Q05 DOUBLE, ST37Q06 DOUBLE, ST37Q07 DOUBLE, ST37Q08 DOUBLE, ST42Q01 DOUBLE, ST42Q02 DOUBLE, ST42Q03 DOUBLE, ST42Q04 DOUBLE, ST42Q05 DOUBLE, ST42Q06 DOUBLE, ST42Q07 DOUBLE, ST42Q08 DOUBLE, ST42Q09 DOUBLE, ST42Q10 DOUBLE, ST43Q01 DOUBLE, ST43Q02 DOUBLE, ST43Q03 DOUBLE, ST43Q04 DOUBLE, ST43Q05 DOUBLE, ST43Q06 DOUBLE, ST44Q01 DOUBLE, ST44Q03 DOUBLE, ST44Q04 DOUBLE, ST44Q05 DOUBLE, ST44Q07 DOUBLE, ST44Q08 DOUBLE, ST46Q01 DOUBLE, ST46Q02 DOUBLE, ST46Q03 DOUBLE, ST46Q04 DOUBLE, ST46Q05 DOUBLE, ST46Q06 DOUBLE, ST46Q07 DOUBLE, ST46Q08 DOUBLE, ST46Q09 DOUBLE, ST48Q01 DOUBLE, ST48Q02 DOUBLE, ST48Q03 DOUBLE, ST48Q04 DOUBLE, ST48Q05 DOUBLE, ST49Q01 DOUBLE, ST49Q02 DOUBLE, ST49Q03 DOUBLE, ST49Q04 DOUBLE, ST49Q05 DOUBLE, ST49Q06 DOUBLE, ST49Q07 DOUBLE, ST49Q09 DOUBLE, ST53Q01 DOUBLE, ST53Q02 DOUBLE, ST53Q03 DOUBLE, ST53Q04 DOUBLE, ST55Q01 DOUBLE, ST55Q02 DOUBLE, ST55Q03 DOUBLE, ST55Q04 DOUBLE, ST57Q01 DOUBLE, ST57Q02 DOUBLE, ST57Q03 DOUBLE, ST57Q04 DOUBLE, ST57Q05 DOUBLE, ST57Q06 DOUBLE, ST61Q01 DOUBLE, ST61Q02 DOUBLE, ST61Q03 DOUBLE, ST61Q04 DOUBLE, ST61Q05 DOUBLE, ST61Q06 DOUBLE, ST61Q07 DOUBLE, ST61Q08 DOUBLE, ST61Q09 DOUBLE, ST62Q01 DOUBLE, ST62Q02 DOUBLE, ST62Q03 DOUBLE, ST62Q04 DOUBLE, ST62Q06 DOUBLE, ST62Q07 DOUBLE, ST62Q08 DOUBLE, ST62Q09 DOUBLE, ST62Q10 DOUBLE, ST62Q11 DOUBLE, ST62Q12 DOUBLE, ST62Q13 DOUBLE, ST62Q15 DOUBLE, ST62Q16 DOUBLE, ST62Q17 DOUBLE, ST62Q19 DOUBLE, ST69Q01 DOUBLE, ST69Q02 DOUBLE, ST69Q03 DOUBLE, ST70Q01 DOUBLE, ST70Q02 DOUBLE, ST70Q03 DOUBLE, ST71Q01 DOUBLE, ST72Q01 DOUBLE, ST73Q01 DOUBLE, ST73Q02 DOUBLE, ST74Q01 DOUBLE, ST74Q02 DOUBLE, ST75Q01 DOUBLE, ST75Q02 DOUBLE, ST76Q01 DOUBLE, ST76Q02 DOUBLE, ST77Q01 DOUBLE, ST77Q02 DOUBLE, ST77Q04 DOUBLE, ST77Q05 DOUBLE, ST77Q06 DOUBLE, ST79Q01 DOUBLE, ST79Q02 DOUBLE, ST79Q03 DOUBLE, ST79Q04 DOUBLE, ST79Q05 DOUBLE, ST79Q06 DOUBLE, ST79Q07 DOUBLE, ST79Q08 DOUBLE, ST79Q10 DOUBLE, ST79Q11 DOUBLE, ST79Q12 DOUBLE, ST79Q15 DOUBLE, ST79Q17 DOUBLE, ST80Q01 DOUBLE, ST80Q04 DOUBLE, ST80Q05 DOUBLE, ST80Q06 DOUBLE, ST80Q07 DOUBLE, ST80Q08 DOUBLE, ST80Q09 DOUBLE, ST80Q10 DOUBLE, ST80Q11 DOUBLE, ST81Q01 DOUBLE, ST81Q02 DOUBLE, ST81Q03 DOUBLE, ST81Q04 DOUBLE, ST81Q05 DOUBLE, ST82Q01 DOUBLE, ST82Q02 DOUBLE, ST82Q03 DOUBLE, ST83Q01 DOUBLE, ST83Q02 DOUBLE, ST83Q03 DOUBLE, ST83Q04 DOUBLE, ST84Q01 DOUBLE, ST84Q02 DOUBLE, ST84Q03 DOUBLE, ST85Q01 DOUBLE, ST85Q02 DOUBLE, ST85Q03 DOUBLE, ST85Q04 DOUBLE, ST86Q01 DOUBLE, ST86Q02 DOUBLE, ST86Q03 DOUBLE, ST86Q04 DOUBLE, ST86Q05 DOUBLE, ST87Q01 DOUBLE, ST87Q02 DOUBLE, ST87Q03 DOUBLE, ST87Q04 DOUBLE, ST87Q05 DOUBLE, ST87Q06 DOUBLE, ST87Q07 DOUBLE, ST87Q08 DOUBLE, ST87Q09 DOUBLE, ST88Q01 DOUBLE, ST88Q02 DOUBLE, ST88Q03 DOUBLE, ST88Q04 DOUBLE, ST89Q02 DOUBLE, ST89Q03 DOUBLE, ST89Q04 DOUBLE, ST89Q05 DOUBLE, ST91Q01 DOUBLE, ST91Q02 DOUBLE, ST91Q03 DOUBLE, ST91Q04 DOUBLE, ST91Q05 DOUBLE, ST91Q06 DOUBLE, ST93Q01 DOUBLE, ST93Q03 DOUBLE, ST93Q04 DOUBLE, ST93Q06 DOUBLE, ST93Q07 DOUBLE, ST94Q05 DOUBLE, ST94Q06 DOUBLE, ST94Q09 DOUBLE, ST94Q10 DOUBLE, ST94Q14 DOUBLE, ST96Q01 DOUBLE, ST96Q02 DOUBLE, ST96Q03 DOUBLE, ST96Q05 DOUBLE, ST101Q01 DOUBLE, ST101Q02 DOUBLE, ST101Q03 DOUBLE, ST101Q05 DOUBLE, ST104Q01 DOUBLE, ST104Q04 DOUBLE, ST104Q05 DOUBLE, ST104Q06 DOUBLE, IC01Q01 DOUBLE, IC01Q02 DOUBLE, IC01Q03 DOUBLE, IC01Q04 DOUBLE, IC01Q05 DOUBLE, IC01Q06 DOUBLE, IC01Q07 DOUBLE, IC01Q08 DOUBLE, IC01Q09 DOUBLE, IC01Q10 DOUBLE, IC01Q11 DOUBLE, IC02Q01 DOUBLE, IC02Q02 DOUBLE, IC02Q03 DOUBLE, IC02Q04 DOUBLE, IC02Q05 DOUBLE, IC02Q06 DOUBLE, IC02Q07 DOUBLE, IC03Q01 DOUBLE, IC04Q01 DOUBLE, IC05Q01 DOUBLE, IC06Q01 DOUBLE, IC07Q01 DOUBLE, IC08Q01 DOUBLE, IC08Q02 DOUBLE, IC08Q03 DOUBLE, IC08Q04 DOUBLE, IC08Q05 DOUBLE, IC08Q06 DOUBLE, IC08Q07 DOUBLE, IC08Q08 DOUBLE, IC08Q09 DOUBLE, IC08Q11 DOUBLE, IC09Q01 DOUBLE, IC09Q02 DOUBLE, IC09Q03 DOUBLE, IC09Q04 DOUBLE, IC09Q05 DOUBLE, IC09Q06 DOUBLE, IC09Q07 DOUBLE, IC10Q01 DOUBLE, IC10Q02 DOUBLE, IC10Q03 DOUBLE, IC10Q04 DOUBLE, IC10Q05 DOUBLE, IC10Q06 DOUBLE, IC10Q07 DOUBLE, IC10Q08 DOUBLE, IC10Q09 DOUBLE, IC11Q01 DOUBLE, IC11Q02 DOUBLE, IC11Q03 DOUBLE, IC11Q04 DOUBLE, IC11Q05 DOUBLE, IC11Q06 DOUBLE, IC11Q07 DOUBLE, IC22Q01 DOUBLE, IC22Q02 DOUBLE, IC22Q04 DOUBLE, IC22Q06 DOUBLE, IC22Q07 DOUBLE, IC22Q08 DOUBLE, EC01Q01 DOUBLE, EC02Q01 DOUBLE, EC03Q01 DOUBLE, EC03Q02 DOUBLE, EC03Q03 DOUBLE, EC03Q04 DOUBLE, EC03Q05 DOUBLE, EC03Q06 DOUBLE, EC03Q07 DOUBLE, EC03Q08 DOUBLE, EC03Q09 DOUBLE, EC03Q10 DOUBLE, EC04Q01A DOUBLE, EC04Q01B DOUBLE, EC04Q01C DOUBLE, EC04Q02A DOUBLE, EC04Q02B DOUBLE, EC04Q02C DOUBLE, EC04Q03A DOUBLE, EC04Q03B DOUBLE, EC04Q03C DOUBLE, EC04Q04A DOUBLE, EC04Q04B DOUBLE, EC04Q04C DOUBLE, EC04Q05A DOUBLE, EC04Q05B DOUBLE, EC04Q05C DOUBLE, EC04Q06A DOUBLE, EC04Q06B DOUBLE, EC04Q06C DOUBLE, EC05Q01 DOUBLE, EC06Q01 DOUBLE, EC07Q01 DOUBLE, EC07Q02 DOUBLE, EC07Q03 DOUBLE, EC07Q04 DOUBLE, EC07Q05 DOUBLE, EC08Q01 DOUBLE, EC08Q02 DOUBLE, EC08Q03 DOUBLE, EC08Q04 DOUBLE, EC09Q03 DOUBLE, EC10Q01 DOUBLE, EC11Q02 DOUBLE, EC11Q03 DOUBLE, EC12Q01 DOUBLE, ST22Q01 DOUBLE, ST23Q01 DOUBLE, ST23Q02 DOUBLE, ST23Q03 DOUBLE, ST23Q04 DOUBLE, ST23Q05 DOUBLE, ST23Q06 DOUBLE, ST23Q07 DOUBLE, ST23Q08 DOUBLE, ST24Q01 DOUBLE, ST24Q02 DOUBLE, ST24Q03 DOUBLE, CLCUSE1 DOUBLE, CLCUSE301 DOUBLE, CLCUSE302 DOUBLE, DEFFORT DOUBLE, QUESTID DOUBLE, BOOKID DOUBLE, EASY DOUBLE, AGE DOUBLE, GRADE DOUBLE, PROGN DOUBLE, ANXMAT DOUBLE, ATSCHL DOUBLE, ATTLNACT DOUBLE, BELONG DOUBLE, BFMJ2 DOUBLE, BMMJ1 DOUBLE, CLSMAN DOUBLE, COBN_F DOUBLE, COBN_M DOUBLE, COBN_S DOUBLE, COGACT DOUBLE, CULTDIST DOUBLE, CULTPOS DOUBLE, DISCLIMA DOUBLE, ENTUSE DOUBLE, ESCS DOUBLE, EXAPPLM DOUBLE, EXPUREM DOUBLE, FAILMAT DOUBLE, FAMCON DOUBLE, FAMCONC DOUBLE, FAMSTRUC DOUBLE, FISCED DOUBLE, HEDRES DOUBLE, HERITCUL DOUBLE, HISCED DOUBLE, HISEI DOUBLE, HOMEPOS DOUBLE, HOMSCH DOUBLE, HOSTCUL DOUBLE, ICTATTNEG DOUBLE, ICTATTPOS DOUBLE, ICTHOME DOUBLE, ICTRES DOUBLE, ICTSCH DOUBLE, IMMIG DOUBLE, INFOCAR DOUBLE, INFOJOB1 DOUBLE, INFOJOB2 DOUBLE, INSTMOT DOUBLE, INTMAT DOUBLE, ISCEDD DOUBLE, ISCEDL DOUBLE, ISCEDO DOUBLE, LANGCOMM DOUBLE, LANGN DOUBLE, LANGRPPD DOUBLE, LMINS DOUBLE, MATBEH DOUBLE, MATHEFF DOUBLE, MATINTFC DOUBLE, MATWKETH DOUBLE, MISCED DOUBLE, MMINS DOUBLE, MTSUP DOUBLE, OCOD1 DOUBLE, OCOD2 DOUBLE, OPENPS DOUBLE, OUTHOURS DOUBLE, PARED DOUBLE, PERSEV DOUBLE, REPEAT DOUBLE, SCMAT DOUBLE, SMINS DOUBLE, STUDREL DOUBLE, SUBNORM DOUBLE, TCHBEHFA DOUBLE, TCHBEHSO DOUBLE, TCHBEHTD DOUBLE, TEACHSUP DOUBLE, TESTLANG DOUBLE, TIMEINT DOUBLE, USEMATH DOUBLE, USESCH DOUBLE, WEALTH DOUBLE, ANCATSCHL DOUBLE, ANCATTLNACT DOUBLE, ANCBELONG DOUBLE, ANCCLSMAN DOUBLE, ANCCOGACT DOUBLE, ANCINSTMOT DOUBLE, ANCINTMAT DOUBLE, ANCMATWKETH DOUBLE, ANCMTSUP DOUBLE, ANCSCMAT DOUBLE, ANCSTUDREL DOUBLE, ANCSUBNORM DOUBLE, PV1MATH DOUBLE, PV2MATH DOUBLE, PV3MATH DOUBLE, PV4MATH DOUBLE, PV5MATH DOUBLE, PV1MACC DOUBLE, PV2MACC DOUBLE, PV3MACC DOUBLE, PV4MACC DOUBLE, PV5MACC DOUBLE, PV1MACQ DOUBLE, PV2MACQ DOUBLE, PV3MACQ DOUBLE, PV4MACQ DOUBLE, PV5MACQ DOUBLE, PV1MACS DOUBLE, PV2MACS DOUBLE, PV3MACS DOUBLE, PV4MACS DOUBLE, PV5MACS DOUBLE, PV1MACU DOUBLE, PV2MACU DOUBLE, PV3MACU DOUBLE, PV4MACU DOUBLE, PV5MACU DOUBLE, PV1MAPE DOUBLE, PV2MAPE DOUBLE, PV3MAPE DOUBLE, PV4MAPE DOUBLE, PV5MAPE DOUBLE, PV1MAPF DOUBLE, PV2MAPF DOUBLE, PV3MAPF DOUBLE, PV4MAPF DOUBLE, PV5MAPF DOUBLE, PV1MAPI DOUBLE, PV2MAPI DOUBLE, PV3MAPI DOUBLE, PV4MAPI DOUBLE, PV5MAPI DOUBLE, PV1READ DOUBLE, PV2READ DOUBLE, PV3READ DOUBLE, PV4READ DOUBLE, PV5READ DOUBLE, PV1SCIE DOUBLE, PV2SCIE DOUBLE, PV3SCIE DOUBLE, PV4SCIE DOUBLE, PV5SCIE DOUBLE, W_FSTUWT DOUBLE, W_FSTR1 DOUBLE, W_FSTR2 DOUBLE, W_FSTR3 DOUBLE, W_FSTR4 DOUBLE, W_FSTR5 DOUBLE, W_FSTR6 DOUBLE, W_FSTR7 DOUBLE, W_FSTR8 DOUBLE, W_FSTR9 DOUBLE, W_FSTR10 DOUBLE, W_FSTR11 DOUBLE, W_FSTR12 DOUBLE, W_FSTR13 DOUBLE, W_FSTR14 DOUBLE, W_FSTR15 DOUBLE, W_FSTR16 DOUBLE, W_FSTR17 DOUBLE, W_FSTR18 DOUBLE, W_FSTR19 DOUBLE, W_FSTR20 DOUBLE, W_FSTR21 DOUBLE, W_FSTR22 DOUBLE, W_FSTR23 DOUBLE, W_FSTR24 DOUBLE, W_FSTR25 DOUBLE, W_FSTR26 DOUBLE, W_FSTR27 DOUBLE, W_FSTR28 DOUBLE, W_FSTR29 DOUBLE, W_FSTR30 DOUBLE, W_FSTR31 DOUBLE, W_FSTR32 DOUBLE, W_FSTR33 DOUBLE, W_FSTR34 DOUBLE, W_FSTR35 DOUBLE, W_FSTR36 DOUBLE, W_FSTR37 DOUBLE, W_FSTR38 DOUBLE, W_FSTR39 DOUBLE, W_FSTR40 DOUBLE, W_FSTR41 DOUBLE, W_FSTR42 DOUBLE, W_FSTR43 DOUBLE, W_FSTR44 DOUBLE, W_FSTR45 DOUBLE, W_FSTR46 DOUBLE, W_FSTR47 DOUBLE, W_FSTR48 DOUBLE, W_FSTR49 DOUBLE, W_FSTR50 DOUBLE, W_FSTR51 DOUBLE, W_FSTR52 DOUBLE, W_FSTR53 DOUBLE, W_FSTR54 DOUBLE, W_FSTR55 DOUBLE, W_FSTR56 DOUBLE, W_FSTR57 DOUBLE, W_FSTR58 DOUBLE, W_FSTR59 DOUBLE, W_FSTR60 DOUBLE, W_FSTR61 DOUBLE, W_FSTR62 DOUBLE, W_FSTR63 DOUBLE, W_FSTR64 DOUBLE, W_FSTR65 DOUBLE, W_FSTR66 DOUBLE, W_FSTR67 DOUBLE, W_FSTR68 DOUBLE, W_FSTR69 DOUBLE, W_FSTR70 DOUBLE, W_FSTR71 DOUBLE, W_FSTR72 DOUBLE, W_FSTR73 DOUBLE, W_FSTR74 DOUBLE, W_FSTR75 DOUBLE, W_FSTR76 DOUBLE, W_FSTR77 DOUBLE, W_FSTR78 DOUBLE, W_FSTR79 DOUBLE, W_FSTR80 DOUBLE, WVARSTRR DOUBLE, VAR_UNIT DOUBLE, SENWGT_STU DOUBLE, VER_STU DOUBLE, PRIMARY KEY (input_id));

LOAD DATA LOCAL INFILE 'pisa2012.csv' INTO TABLE udacity.pisa2012 FIELDS TERMINATED BY "," (CNT, SUBNATIO, STRATUM, OECD, NC, SCHOOLID, STIDSTD, ST01Q01, ST02Q01, ST03Q01, ST03Q02, ST04Q01, ST05Q01, ST06Q01, ST07Q01, ST07Q02, ST07Q03, ST08Q01, ST09Q01, ST115Q01, ST11Q01, ST11Q02, ST11Q03, ST11Q04, ST11Q05, ST11Q06, ST13Q01, ST14Q01, ST14Q02, ST14Q03, ST14Q04, ST15Q01, ST17Q01, ST18Q01, ST18Q02, ST18Q03, ST18Q04, ST19Q01, ST20Q01, ST20Q02, ST20Q03, ST21Q01, ST25Q01, ST26Q01, ST26Q02, ST26Q03, ST26Q04, ST26Q05, ST26Q06, ST26Q07, ST26Q08, ST26Q09, ST26Q10, ST26Q11, ST26Q12, ST26Q13, ST26Q14, ST26Q15, ST26Q16, ST26Q17, ST27Q01, ST27Q02, ST27Q03, ST27Q04, ST27Q05, ST28Q01, ST29Q01, ST29Q02, ST29Q03, ST29Q04, ST29Q05, ST29Q06, ST29Q07, ST29Q08, ST35Q01, ST35Q02, ST35Q03, ST35Q04, ST35Q05, ST35Q06, ST37Q01, ST37Q02, ST37Q03, ST37Q04, ST37Q05, ST37Q06, ST37Q07, ST37Q08, ST42Q01, ST42Q02, ST42Q03, ST42Q04, ST42Q05, ST42Q06, ST42Q07, ST42Q08, ST42Q09, ST42Q10, ST43Q01, ST43Q02, ST43Q03, ST43Q04, ST43Q05, ST43Q06, ST44Q01, ST44Q03, ST44Q04, ST44Q05, ST44Q07, ST44Q08, ST46Q01, ST46Q02, ST46Q03, ST46Q04, ST46Q05, ST46Q06, ST46Q07, ST46Q08, ST46Q09, ST48Q01, ST48Q02, ST48Q03, ST48Q04, ST48Q05, ST49Q01, ST49Q02, ST49Q03, ST49Q04, ST49Q05, ST49Q06, ST49Q07, ST49Q09, ST53Q01, ST53Q02, ST53Q03, ST53Q04, ST55Q01, ST55Q02, ST55Q03, ST55Q04, ST57Q01, ST57Q02, ST57Q03, ST57Q04, ST57Q05, ST57Q06, ST61Q01, ST61Q02, ST61Q03, ST61Q04, ST61Q05, ST61Q06, ST61Q07, ST61Q08, ST61Q09, ST62Q01, ST62Q02, ST62Q03, ST62Q04, ST62Q06, ST62Q07, ST62Q08, ST62Q09, ST62Q10, ST62Q11, ST62Q12, ST62Q13, ST62Q15, ST62Q16, ST62Q17, ST62Q19, ST69Q01, ST69Q02, ST69Q03, ST70Q01, ST70Q02, ST70Q03, ST71Q01, ST72Q01, ST73Q01, ST73Q02, ST74Q01, ST74Q02, ST75Q01, ST75Q02, ST76Q01, ST76Q02, ST77Q01, ST77Q02, ST77Q04, ST77Q05, ST77Q06, ST79Q01, ST79Q02, ST79Q03, ST79Q04, ST79Q05, ST79Q06, ST79Q07, ST79Q08, ST79Q10, ST79Q11, ST79Q12, ST79Q15, ST79Q17, ST80Q01, ST80Q04, ST80Q05, ST80Q06, ST80Q07, ST80Q08, ST80Q09, ST80Q10, ST80Q11, ST81Q01, ST81Q02, ST81Q03, ST81Q04, ST81Q05, ST82Q01, ST82Q02, ST82Q03, ST83Q01, ST83Q02, ST83Q03, ST83Q04, ST84Q01, ST84Q02, ST84Q03, ST85Q01, ST85Q02, ST85Q03, ST85Q04, ST86Q01, ST86Q02, ST86Q03, ST86Q04, ST86Q05, ST87Q01, ST87Q02, ST87Q03, ST87Q04, ST87Q05, ST87Q06, ST87Q07, ST87Q08, ST87Q09, ST88Q01, ST88Q02, ST88Q03, ST88Q04, ST89Q02, ST89Q03, ST89Q04, ST89Q05, ST91Q01, ST91Q02, ST91Q03, ST91Q04, ST91Q05, ST91Q06, ST93Q01, ST93Q03, ST93Q04, ST93Q06, ST93Q07, ST94Q05, ST94Q06, ST94Q09, ST94Q10, ST94Q14, ST96Q01, ST96Q02, ST96Q03, ST96Q05, ST101Q01, ST101Q02, ST101Q03, ST101Q05, ST104Q01, ST104Q04, ST104Q05, ST104Q06, IC01Q01, IC01Q02, IC01Q03, IC01Q04, IC01Q05, IC01Q06, IC01Q07, IC01Q08, IC01Q09, IC01Q10, IC01Q11, IC02Q01, IC02Q02, IC02Q03, IC02Q04, IC02Q05, IC02Q06, IC02Q07, IC03Q01, IC04Q01, IC05Q01, IC06Q01, IC07Q01, IC08Q01, IC08Q02, IC08Q03, IC08Q04, IC08Q05, IC08Q06, IC08Q07, IC08Q08, IC08Q09, IC08Q11, IC09Q01, IC09Q02, IC09Q03, IC09Q04, IC09Q05, IC09Q06, IC09Q07, IC10Q01, IC10Q02, IC10Q03, IC10Q04, IC10Q05, IC10Q06, IC10Q07, IC10Q08, IC10Q09, IC11Q01, IC11Q02, IC11Q03, IC11Q04, IC11Q05, IC11Q06, IC11Q07, IC22Q01, IC22Q02, IC22Q04, IC22Q06, IC22Q07, IC22Q08, EC01Q01, EC02Q01, EC03Q01, EC03Q02, EC03Q03, EC03Q04, EC03Q05, EC03Q06, EC03Q07, EC03Q08, EC03Q09, EC03Q10, EC04Q01A, EC04Q01B, EC04Q01C, EC04Q02A, EC04Q02B, EC04Q02C, EC04Q03A, EC04Q03B, EC04Q03C, EC04Q04A, EC04Q04B, EC04Q04C, EC04Q05A, EC04Q05B, EC04Q05C, EC04Q06A, EC04Q06B, EC04Q06C, EC05Q01, EC06Q01, EC07Q01, EC07Q02, EC07Q03, EC07Q04, EC07Q05, EC08Q01, EC08Q02, EC08Q03, EC08Q04, EC09Q03, EC10Q01, EC11Q02, EC11Q03, EC12Q01, ST22Q01, ST23Q01, ST23Q02, ST23Q03, ST23Q04, ST23Q05, ST23Q06, ST23Q07, ST23Q08, ST24Q01, ST24Q02, ST24Q03, CLCUSE1, CLCUSE301, CLCUSE302, DEFFORT, QUESTID, BOOKID, EASY, AGE, GRADE, PROGN, ANXMAT, ATSCHL, ATTLNACT, BELONG, BFMJ2, BMMJ1, CLSMAN, COBN_F, COBN_M, COBN_S, COGACT, CULTDIST, CULTPOS, DISCLIMA, ENTUSE, ESCS, EXAPPLM, EXPUREM, FAILMAT, FAMCON, FAMCONC, FAMSTRUC, FISCED, HEDRES, HERITCUL, HISCED, HISEI, HOMEPOS, HOMSCH, HOSTCUL, ICTATTNEG, ICTATTPOS, ICTHOME, ICTRES, ICTSCH, IMMIG, INFOCAR, INFOJOB1, INFOJOB2, INSTMOT, INTMAT, ISCEDD, ISCEDL, ISCEDO, LANGCOMM, LANGN, LANGRPPD, LMINS, MATBEH, MATHEFF, MATINTFC, MATWKETH, MISCED, MMINS, MTSUP, OCOD1, OCOD2, OPENPS, OUTHOURS, PARED, PERSEV, REPEAT, SCMAT, SMINS, STUDREL, SUBNORM, TCHBEHFA, TCHBEHSO, TCHBEHTD, TEACHSUP, TESTLANG, TIMEINT, USEMATH, USESCH, WEALTH, ANCATSCHL, ANCATTLNACT, ANCBELONG, ANCCLSMAN, ANCCOGACT, ANCINSTMOT, ANCINTMAT, ANCMATWKETH, ANCMTSUP, ANCSCMAT, ANCSTUDREL, ANCSUBNORM, PV1MATH, PV2MATH, PV3MATH, PV4MATH, PV5MATH, PV1MACC, PV2MACC, PV3MACC, PV4MACC, PV5MACC, PV1MACQ, PV2MACQ, PV3MACQ, PV4MACQ, PV5MACQ, PV1MACS, PV2MACS, PV3MACS, PV4MACS, PV5MACS, PV1MACU, PV2MACU, PV3MACU, PV4MACU, PV5MACU, PV1MAPE, PV2MAPE, PV3MAPE, PV4MAPE, PV5MAPE, PV1MAPF, PV2MAPF, PV3MAPF, PV4MAPF, PV5MAPF, PV1MAPI, PV2MAPI, PV3MAPI, PV4MAPI, PV5MAPI, PV1READ, PV2READ, PV3READ, PV4READ, PV5READ, PV1SCIE, PV2SCIE, PV3SCIE, PV4SCIE, PV5SCIE, W_FSTUWT, W_FSTR1, W_FSTR2, W_FSTR3, W_FSTR4, W_FSTR5, W_FSTR6, W_FSTR7, W_FSTR8, W_FSTR9, W_FSTR10, W_FSTR11, W_FSTR12, W_FSTR13, W_FSTR14, W_FSTR15, W_FSTR16, W_FSTR17, W_FSTR18, W_FSTR19, W_FSTR20, W_FSTR21, W_FSTR22, W_FSTR23, W_FSTR24, W_FSTR25, W_FSTR26, W_FSTR27, W_FSTR28, W_FSTR29, W_FSTR30, W_FSTR31, W_FSTR32, W_FSTR33, W_FSTR34, W_FSTR35, W_FSTR36, W_FSTR37, W_FSTR38, W_FSTR39, W_FSTR40, W_FSTR41, W_FSTR42, W_FSTR43, W_FSTR44, W_FSTR45, W_FSTR46, W_FSTR47, W_FSTR48, W_FSTR49, W_FSTR50, W_FSTR51, W_FSTR52, W_FSTR53, W_FSTR54, W_FSTR55, W_FSTR56, W_FSTR57, W_FSTR58, W_FSTR59, W_FSTR60, W_FSTR61, W_FSTR62, W_FSTR63, W_FSTR64, W_FSTR65, W_FSTR66, W_FSTR67, W_FSTR68, W_FSTR69, W_FSTR70, W_FSTR71, W_FSTR72, W_FSTR73, W_FSTR74, W_FSTR75, W_FSTR76, W_FSTR77, W_FSTR78, W_FSTR79, W_FSTR80, WVARSTRR, VAR_UNIT, SENWGT_STU, VER_STU);

ALTER TABLE udacity.pisa2012 ADD INDEX CNT(CNT);
ALTER TABLE udacity.pisa2012 ADD INDEX ST01Q01(ST01Q01);
ALTER TABLE udacity.pisa2012 ADD INDEX ST03Q02(ST03Q02);
ALTER TABLE udacity.pisa2012 ADD INDEX ST03Q03(ST03Q03);
ALTER TABLE udacity.pisa2012 ADD INDEX ST04Q01(ST04Q01);
ALTER TABLE udacity.pisa2012 ADD INDEX ST06Q01(ST06Q01);
ALTER TABLE udacity.pisa2012 ADD INDEX WEALTH(WEALTH);
ALTER TABLE udacity.pisa2012 ADD INDEX PV1MATH(PV1MATH);
ALTER TABLE udacity.pisa2012 ADD INDEX PV2MATH(PV2MATH);
ALTER TABLE udacity.pisa2012 ADD INDEX PV3MATH(PV3MATH);
ALTER TABLE udacity.pisa2012 ADD INDEX PV4MATH(PV4MATH);
ALTER TABLE udacity.pisa2012 ADD INDEX PV5MATH(PV5MATH);
ALTER TABLE udacity.pisa2012 ADD INDEX PV1READ(PV1READ);
ALTER TABLE udacity.pisa2012 ADD INDEX PV2READ(PV2READ);
ALTER TABLE udacity.pisa2012 ADD INDEX PV3READ(PV3READ);
ALTER TABLE udacity.pisa2012 ADD INDEX PV4READ(PV4READ);
ALTER TABLE udacity.pisa2012 ADD INDEX PV5READ(PV5READ);
ALTER TABLE udacity.pisa2012 ADD INDEX PV1SCIE(PV1SCIE);
ALTER TABLE udacity.pisa2012 ADD INDEX PV2SCIE(PV2SCIE);
ALTER TABLE udacity.pisa2012 ADD INDEX PV3SCIE(PV3SCIE);
ALTER TABLE udacity.pisa2012 ADD INDEX PV4SCIE(PV4SCIE);
ALTER TABLE udacity.pisa2012 ADD INDEX PV5SCIE(PV5SCIE);
