#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.6
import json

from draft.entityRelatedLangConf import LangSet

__author__ = 'Lou Zehua'
__time__ = '2018/9/30 14:31'

URI_PREFIX = 'http://thunisoft.org/item/'
entity_URI = ''  # contains the value of URI_PREFIX and the value of its KgId: e.g. "http://thunisoft.org/item/Q1"

# The format of entity knowledge: a init for entities with the format like example in file: entityFormatExample.json
# entity Identifier
KEY_KGID = 'KgId'  # variable but should be the KgId value stored in database: e.g. "Q1"

# URI: a link of the online corpus: e.g. "https://baike.baidu.com/item/腰部/29900"
KEY_URI = 'URI'

# language: the language set contains the entity
KEY_LANGUAGE = 'language'
LANGUAGE_DEFAULT = LangSet.ZH

# mention: the default character form of the entity
KEY_MENTION = 'mention'

# aliases
KEY_ALIASES = 'aliases'

# claims
KEY_CLAIMS = 'claims'

# description
KEY_DESCRIPTION = 'description'

# mainsnap
KEY_MAINSNAP = 'mainsnap'

# key prefix of snapshot level
KEY_PREFIX_SNAPSHOT_LEVEL = 'level'


# alias: e.g. language="en-us", _alias_str_List=["left","a surname","wrong","assist"]
class Alias:
    def __init__(self, language):
        self._alias_str_List = []
        self.language = ''
        if type(language) == str and language in LangSet().__VARS__:
            self.language = language
            self.setAliasStrList()

    def setAliasStrList(self, aliasStrList=None, override=True):
        if self.language and aliasStrList and type(aliasStrList) == list:
            for aliasStr in aliasStrList:
                if type(aliasStr) != str:
                    return None
            if override:
                self._alias_str_List = aliasStrList
            else:
                self._alias_str_List.extend(aliasStrList)

    def getAliasStrList(self):
        return self._alias_str_List


# aliases: e.g. languageList=["zh-cn","en-us"]
# _aliases_Dict={
#     "zh-cn": ["佐"],
#     "en-us": ["left","a surname","wrong","assist"]
# }
class Aliases:
    def __init__(self):
        self._aliases_Dict = {}
        self.setAliasesDict()

    def setAliasesDict(self, aliasObjList=None, override=True):
        if aliasObjList and type(aliasObjList) == list:
            for alias in aliasObjList:
                language = alias.language
                aliasStrList = alias.getAliasStrList()
                if isinstance(alias, Alias) and language and aliasStrList:
                    if override:
                        self._aliases_Dict[language] = aliasStrList
                    else:
                        if language not in self._aliases_Dict.keys():
                            self._aliases_Dict[language] = []
                        self._aliases_Dict[language].extend(aliasStrList)

    def getAliasesDict(self):
        return self._aliases_Dict

# SnapRecord
# {"level1":"左边","language":"zh-cn"}
class SnapRecord:
    def __init__(self, level_Key, level_Value, language):
        if type(level_Key) == str and type(level_Value) == str and type(language) == str and language in LangSet().__VARS__:
            self._snapRecord_Dict = {}
            self.level_Key = level_Key
            self.level_Value = level_Value
            self.language = language
            self.setSnapRecordDict()

    def setSnapRecordDict(self):
        self._snapRecord_Dict.clear()
        self._snapRecord_Dict[KEY_LANGUAGE] = self.language
        self._snapRecord_Dict[self.level_Key] = self.level_Value

    def getSnapRecordDict(self):
        return self._snapRecord_Dict


# mainsnap: e.g.
# [
#     {"level1":"汉字","language":"zh-cn"},
#     {"level2":"方位","language":"zh-cn"}
# ]
class Mainsnap:
    def __init__(self):
        self._mainsnap_DataList = []
        self.setMainsnap_DataList()

    def setMainsnap_DataList(self, snapRecordList=None, override=True):
        if snapRecordList and type(snapRecordList) == list:
            for snapRecord in snapRecordList:
                if not isinstance(snapRecord, SnapRecord):
                    return None
            if override:
                self._mainsnap_DataList = [snapRecord.getSnapRecordDict() for snapRecord in snapRecordList]
            else:
                self._mainsnap_DataList.extend([snapRecord.getSnapRecordDict() for snapRecord in snapRecordList])

    def getMainsnap_DataList(self):
        return self._mainsnap_DataList


# claim: e.g.
# {propOperation: _claim_Dict}
# propOperation = "P1549"
# _claim_Dict = {
#     "mainsnap":[
#         {"level1":"左边","language":"zh-cn"}
#     ]
# }
class Claim:
    def __init__(self, propOperation):
        self._claim_Dict = {}
        self.propOperation = ''
        if type(propOperation) == str:
            self.propOperation = propOperation
            self.setClaimDict()

    def setClaimDict(self, mainsnapObj=None):
        if isinstance(mainsnapObj, Mainsnap):
            Mainsnap_DataList = mainsnapObj.getMainsnap_DataList()
            if Mainsnap_DataList:
                self._claim_Dict[KEY_MAINSNAP] = Mainsnap_DataList

    def getClaimDict(self):
        return self._claim_Dict


# claims: e.g.
# _claims_Dict = {
#     "P1549": {
#         "mainsnap":[
#             {"level1":"左边","language":"zh-cn"}
#         ]
#     },
#     "superClass": {
#         "mainsnap":[
#             {"level1":"汉字","language":"zh-cn"},
#             {"level2":"方位","language":"zh-cn"}
#         ]
#     }
# }
class Claims:
    def __init__(self):
        self._claims_Dict = {}
        self.setClaimsDict()

    def setClaimsDict(self, claimObjList=None):
        if claimObjList and type(claimObjList) == list:
            for claim in claimObjList:
                propOperation = claim.propOperation
                claimDict = claim.getClaimDict()
                if isinstance(claim, Claim) and propOperation and claimDict:
                    self._claims_Dict[propOperation] = claimDict

    def getClaimsDict(self):
        return self._claims_Dict


# class entity: e.g.
# {
#     "http://thunisoft.org/item/Q1": {
#         "KgId": "Q1",
#         "URI": "https://baike.baidu.com/item/腰部/29900",
#         "language": "zh-cn",
#         "mention": "左",
#         "aliases": {
#             "zh-cn": ["佐"],
#             "en-us": ["left","a surname","wrong","assist"]
#         },
#         "claims": {
#             "P1549": {
#                 "mainsnap":[
#                     {"level1":"左边","language":"zh-cn"}
#                 ]
#             },
#             "superClass": {
#                 "mainsnap":[
#                     {"level1":"汉字","language":"zh-cn"},
#                     {"level2":"方位","language":"zh-cn"}
# 		        ]
#             }
#         },
#         "desc": "汉字"
#     }
# }
class Entity:
    def __init__(self, KgId=None, URI=None, mention=None, aliases=None, claims=None, description=None,
                 language=LANGUAGE_DEFAULT):
        self._entity_Dict = {}
        self.aliases_Dcit = {}
        self.claims_Dcit = {}
        self.KgId = KgId if self.strVlidate(KgId) else ''
        self.URI = URI if self.strVlidate(URI) else ''
        self.language = language
        self.mention = mention if self.strVlidate(mention) else ''
        self.aliases = aliases if aliases else {}
        self.claims = claims if claims else {}
        self.description = description if self.strVlidate(description) else ''
        self.setEntityDict(self.aliases, self.claims)

    def strVlidate(self, strParam):
        return strParam and type(strParam) == str

    def setEntityDict(self, aliasesObj=None, claimsObj=None):
        if not self.KgId:
            return None
        if aliasesObj and isinstance(aliasesObj, Aliases):
            self.aliases_Dcit = aliasesObj.getAliasesDict()
        if claimsObj and isinstance(claimsObj, Claims):
            self.claims_Dcit = claimsObj.getClaimsDict()
        self._entity_Dict = {
            KEY_KGID: self.KgId,
            KEY_URI: self.URI,
            KEY_LANGUAGE: self.language,
            KEY_MENTION: self.mention,
            KEY_ALIASES: self.aliases_Dcit,
            KEY_CLAIMS: self.claims_Dcit,
            KEY_DESCRIPTION: self.description
        }

    def getEntityDict(self):
        return self._entity_Dict

    def getEntityLocURI(self, relative=False):
        if not self.KgId:
            return None
        return self.KgId if relative else URI_PREFIX + self.KgId

    def loadFromEntityDict(self, entityDict):
        try:
            # create alias
            aliasesDict = entityDict[KEY_ALIASES]
            aliase_List = []
            for k_alias, v_alias in aliasesDict.items():
                alias = Alias(language=k_alias)
                alias.setAliasStrList(v_alias)
                aliase_List.append(alias)
            # generate aliases
            aliases = Aliases()
            aliases.setAliasesDict(aliase_List)

            # create claim
            claimsDict = entityDict[KEY_CLAIMS]
            claimDict_List = []
            for k_claim, v_claim in claimsDict.items():
                mainsnapData = v_claim[KEY_MAINSNAP]
                snapRecord_List = []
                for snapRecordData in mainsnapData:
                    level_Key_snapRecord = ''
                    level_Value_snapRecord = ''
                    language_snapRecord = ''
                    for k_snapRecord in snapRecordData.keys():
                        if k_snapRecord == KEY_LANGUAGE:
                            language_snapRecord = snapRecordData[k_snapRecord]
                        elif str(k_snapRecord).startswith(KEY_PREFIX_SNAPSHOT_LEVEL):
                            level_Key_snapRecord = k_snapRecord
                            level_Value_snapRecord = snapRecordData[k_snapRecord]
                    snapRecord = SnapRecord(level_Key=level_Key_snapRecord, level_Value=level_Value_snapRecord, language=language_snapRecord)
                    snapRecord_List.append(snapRecord)
                claim = Claim(propOperation=k_claim)
                mainsnap = Mainsnap()
                mainsnap.setMainsnap_DataList(snapRecord_List)
                claim.setClaimDict(mainsnap)
                claimDict_List.append(claim)
            # generate claims
            claims = Claims()
            claims.setClaimsDict(claimDict_List)
            entity = Entity(KgId=entityDict[KEY_KGID], URI=entityDict[KEY_URI], mention=entityDict[KEY_MENTION],
                            aliases=aliases, claims=claims, description=entityDict[KEY_DESCRIPTION])
            return entity
        except BaseException:
            return None


def main():

    # create alias
    alias1 = Alias(language=LangSet.ZH)
    alias1.setAliasStrList(['佐'])
    alias2 = Alias(language=LangSet.EN)
    alias2.setAliasStrList(['left', 'a surname', 'wrong', 'assist'])
    # generate aliases
    aliases = Aliases()
    aliases.setAliasesDict([alias1,alias2])

    # create claim
    snapRecord1 = SnapRecord(level_Key='level1', level_Value='汉字', language=LangSet.ZH)
    snapRecord2 = SnapRecord(level_Key='level2', level_Value='方位', language=LangSet.ZH)
    mainsnap1 = Mainsnap()
    mainsnap1.setMainsnap_DataList([snapRecord1, snapRecord2])
    mainsnap2 = Mainsnap()
    mainsnap2.setMainsnap_DataList([snapRecord1])
    claim1 = Claim('P1')
    claim1.setClaimDict(mainsnap1)
    claim2 = Claim('P2')
    claim2.setClaimDict(mainsnap2)
    # generate claims
    claims = Claims()
    claims.setClaimsDict([claim1, claim2])

    # merge to entity
    entity = Entity(KgId='Q1', URI='http://ex.org/item/xxx', mention='左', aliases=aliases, claims=claims, description='汉字')
    print('getEntityLocURI', entity.getEntityLocURI(relative=False))
    # entity.setEntityDict(aliasesObj=aliases, claimsObj=claims)
    entityDict = entity.getEntityDict()
    # json form
    entityDictJson = json.dumps(entityDict)
    print('entityJson', entityDictJson)
    entityDictReloaded = json.loads(entityDictJson)
    print('entityDictReloaded', entityDictReloaded)
    print('aliasesDictReloaded', entityDictReloaded[KEY_ALIASES])
    print('claimsDictReloaded', entityDictReloaded[KEY_CLAIMS])
    entityReloaded = Entity()
    entityReloaded = entityReloaded.loadFromEntityDict(entityDictReloaded)
    print('entityReloaded', entityReloaded.__class__)
    print(entityReloaded.__dict__)


if __name__ == '__main__':
    main()