import pandas as pd
import csv
import requests
import tqdm
import json
import os
from elsapy.elsclient import ElsClient
from elsapy.elsprofile import ElsAuthor, ElsAffil
from elsapy.elsdoc import FullDoc, AbsDoc
from elsapy.elssearch import ElsSearch
import pycountry
import re
from urllib.parse import urlparse, parse_qs



def getElsevierAPIKey():
    with open("apiKey.json") as f:
        data = json.load(f)
        return data["apiKey"]

def load_data_csv(file_path):
    # Create a new Dataframe
    data = []
    
    with open(file_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # Parse JSON strings
            json_data_1 = json.loads(row[0])
            json_data_2 = json.loads(row[1])
            json_data_3 = json.loads(row[2])
            
            # Append the data to the list
            data.append([json_data_1, json_data_2, json_data_3])
    
    # Filter the data in which json_data_3 == 1
    data = [d for d in data if d[2] == 1]
    return data


def convert_to_dataframe(data):
    # Create a new Dataframe
    df = pd.DataFrame(data, columns=["Published", "Preprint"])
    return df
        

def fetch_crossref_data(doi):
    url = f"https://api.crossref.org/works/{doi}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(response)
        print(f"Error fetching data for DOI: {doi}")
        return None


def basic_dataset_creation(df):
    data_frame = pd.DataFrame(columns=["SubmissionID", "SubmissionYear", "SubmissionTitle", "SubmissionAbstract", "firstName", "middleName", "lastName", "isFirstAuthor", "city", "country", "countryCode", "institution", "doi", "authorScopusId", "authorPublicationHistory"])
    for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
        
        preprint = row["Preprint"]
        published = row["Published"]
        
        
        submissionID = preprint.get("DOI", None)
        
        date = preprint.get("issued", None)
        if date != None:
            submissionYear = date["date-parts"][0][0]
        else:
            submissionYear = None
            
        submissionTitle = preprint.get("title", None)
        submissionTitle = submissionTitle[0]
        submissionAbstract = None
        city = None
        country = None
        countryCode = None
        institution = None
        doi = published.get("DOI", None)
        
        authors = preprint.get("author", [])
        for author in authors:
            firstName = author.get("given", None)
            middleName = None
            lastName = author.get("family", None)
            isFirstAuthor = None
            authorScopusId = None
            authorPublicationHistory = None
            
            if firstName == None or lastName == None:
                break
            
            new_row = pd.DataFrame([[submissionID, submissionYear, submissionTitle, submissionAbstract, firstName, middleName, lastName, isFirstAuthor, city, country, countryCode, institution, doi, authorScopusId, authorPublicationHistory]], columns=["SubmissionID", "SubmissionYear", "SubmissionTitle", "SubmissionAbstract", "firstName", "middleName", "lastName", "isFirstAuthor", "city", "country", "countryCode", "institution", "doi", "authorScopusId", "authorPublicationHistory"])
            data_frame = pd.concat([data_frame, new_row], ignore_index=True)

    return data_frame


def clean_abstract(abstract):
    # Remove <jats:title> and </jats:title> tags and any content between them
    abstract = re.sub(r'<jats:title>[^>]+</jats:title>', '', abstract)
    # Remove <jats:p> and </jats:p> tags
    abstract = re.sub(r'<jats:p>|</jats:p>', '', abstract)
    # Remove any other <jats:*> tags
    abstract = re.sub(r'<jats:[^>]+>', '', abstract)
    # Remove </jats:sec> tags
    abstract = re.sub(r'</jats:sec>', '', abstract)
    # Remove all other HTML tags
    abstract = re.sub(r'<[^>]+>', '', abstract)
    return abstract

def fetch_crossref_data(doi):
    url = f"https://api.crossref.org/works/{doi}"
    #print(url)
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"Error fetching data for DOI: {doi}")
        return None
    

def extend_dataset_with_crossref(submissionIds):
    listOfIdAbstract = {}
    
    print(submissionIds[:10])
    print(len(submissionIds))
    i = 0
    
    for id in tqdm.tqdm(submissionIds):
        data = fetch_crossref_data(id)
        if data:
            abstract = data.get("message", {}).get("abstract", None)
            if abstract:
                abstract = clean_abstract(abstract)
                listOfIdAbstract[id] = abstract
            else:
                listOfIdAbstract[id] = None
                
    return listOfIdAbstract

def fetch_openalex_data(doi):
    url = f"https://api.openalex.org/works/https://doi.org/{doi}"
    response = requests.get(url)
    if response.status_code == 200:
        #print(response.json())
        return response.json()
    else:
        print(response)
        print(f"Error fetching data for DOI: {doi}")
        return None

def get_country_name(country_code):
    try:
        country = pycountry.countries.get(alpha_2=country_code)
        return country.name
    except:
        return None


def extend_dataset_with_openAlex(df):
    openalex_data = []
    missedOut = 0
    inside = 0
    out = 0
    for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
        doi = row["PreprintID"]
        pubdoi = row["PublicationID"]
        match = row["Match"]
        temp = doi
        doi = pubdoi
        pubdoi = temp
        data = fetch_openalex_data(doi)
        if data:
            language = data.get("language", None)
            if language == "en":
                break_outer_loop = False
                inside_data = []
                submissionID = data.get("doi", None)
                
                submissionID = submissionID[15:]
                submissionYear = data.get("publication_year")
                submissionTitle = data.get("title", None)
                authors = data.get("authorships", [])
                if (len(authors) != 0):
                    
                    
                    for author in authors:
                        notFound = True
                        fullName = author.get("author", {}).get("display_name", None)
                        if (len(fullName.split()) == 2):
                            firstName, lastName = fullName.split()
                            middleName = None
                        elif (len(fullName.split()) == 3):
                            firstName, middleName, lastName = fullName.split()
                        else:
                            firstName = fullName
                            middleName = None
                            lastName = None
                        isFirstAuthor = author.get("author_position", None)
                        if isFirstAuthor == "first":
                            isFirstAuthor = 1
                        else:
                            isFirstAuthor = 0
                        if len(author.get("institutions", [])) != 0:
                            countryCode = author.get("institutions", [])[0].get("country_code", None)
                            country = get_country_name(countryCode) if countryCode else None
                            institution = author.get("institutions", [])[0].get("display_name", None)
                        else:
                            #print("No institutions")
                            break_outer_loop = True
                            break
                        doi = pubdoi
                        if None in [submissionID, submissionYear, submissionTitle, firstName, lastName, isFirstAuthor, country, countryCode, institution, doi]:
                            #print(f"Breaking at index {index} due to missing values")
                            #print(f"submissionID: {submissionID}, submissionYear: {submissionYear}, submissionTitle: {submissionTitle}, firstName: {firstName}, lastName: {lastName}, isFirstAuthor: {isFirstAuthor}, country: {country}, countryCode: {countryCode}, institution: {institution}, doi: {doi}")
                            break_outer_loop = True
                            break
                        
                        authorId = author.get("author", {}).get("id", None)
                        # Extract id from https://openalex.org/A5100341522
                        if authorId:
                            authorId = authorId[21:]
                            url = f"https://api.openalex.org/authors/{authorId}"
                            response = requests.get(url)
                            if response.status_code == 200:
                                authorData = response.json()
                                authorScopusId = authorData.get("ids", {}).get("scopus", None)
                                authorOrchidId = authorData.get("ids", {}).get("orcid", None)
                                
                                    
                                if authorScopusId != None or authorOrchidId != None:
                                    # Extract id from http://www.scopus.com/inward/authorDetails.url?authorID=36455008000&partnerID=MN8TOARS
                                    if authorScopusId != None:
                                        url = authorScopusId
                                        parsed_url = urlparse(url)
                                        # Extract the query parameters
                                        query_params = parse_qs(parsed_url.query)
                                        
                                        

                                        # Get the authorID
                                        authorIDReal = query_params.get("authorID", [None])[0]
                                        
                                        authorIDReal = re.findall(r'\d+', url)
                                        authorIDReal = authorIDReal[0]                                    
                                        scopusURL = "https://api.elsevier.com/content/search/scopus?query=AU-ID(" + authorIDReal + ")&view=COMPLETE"
                                    
                                        headers = {
                                            "X-ELS-APIKey": getElsevierAPIKey(),
                                            "Accept": "application/json"
                                        }
                                        
                                        response = requests.get(scopusURL, headers=headers)
                                        
                                        if response.status_code == 200:

                                            response = response.json()
                                            
                                            authorPublicationHistory = []
                                            log = response.get("search-results", {}).get("entry", None)
                                            if log == None:
                                                authorPublicationHistory = None
                                                out += 1
                                                break
                                            
                                            for paper in log:
                                                pTitle = paper.get("dc:title", None)
                                                pAbstract = paper.get("dc:description", None)
                                                pDOI = paper.get("prism:doi", None)
                                                pCitationCount = paper.get("citedby-count", None)
                                                pScopusID = paper.get("dc:identifier", None)
                                                pAuthors = paper.get("author", None)
                                                pA = []
                                                if pAuthors:
                                                    for author in pAuthors:
                                                        f = author.get("given-name", None)
                                                        l = author.get("surname", None)
                                                        if f != None and l != None:
                                                            a = f + " " + l
                                                            pA.append(a)
                                                authorPublicationHistory.append({"title": pTitle, "abstract": pAbstract, "doi": pDOI, "citationCount": pCitationCount, "scopusID": pScopusID, "authors": pA})
                                            
                                        else:
                                            print("Error fetching data for DOI: ", doi)
                                            break
                                    else:
                                        # url = https://orcid.org/0000-0002-7864-2578'
                                        authorIDReal = authorOrchidId[18:]
                                        scopusURL = "https://api.elsevier.com/content/search/scopus?query=orcid(" + authorIDReal + ")&view=COMPLETE"
                                    
                                        headers = {
                                            "X-ELS-APIKey": getElsevierAPIKey(),
                                            "Accept": "application/json"
                                        }
                                        
                                        response = requests.get(scopusURL, headers=headers)
                                        
                                        if response.status_code == 200:
                                            response = response.json()
                                            
                                            authorPublicationHistory = []
                                            log = response.get("search-results", {}).get("entry", None)
                                            if log == None:
                                                authorPublicationHistory = None
                                                out += 1
                                                break
                                            
                                            for paper in log:
                                                pTitle = paper.get("dc:title", None)
                                                pAbstract = paper.get("dc:description", None)
                                                pDOI = paper.get("prism:doi", None)
                                                pCitationCount = paper.get("citedby-count", None)
                                                pScopusID = paper.get("dc:identifier", None)
                                                pAuthors = paper.get("author", None)
                                                pA = []
                                                if pAuthors:
                                                    for author in pAuthors:
                                                        f = author.get("given-name", None)
                                                        l = author.get("surname", None)
                                                        if f != None and l != None:
                                                            
                                                            a = f + " " + l
                                                            pA.append(a)
                                                authorPublicationHistory.append({"title": pTitle, "abstract": pAbstract, "doi": pDOI, "citationCount": pCitationCount, "scopusID": pScopusID, "authors": pA})
                                        else:
                                            print("Error fetching data for DOI: ", doi)
                                            break
                                    
                                    
                                else:
                                    #print("No Scopus ID")
                                    #print(authorOrchidId)
                                    #print(authorData.get("ids", {}))
                                    authorScopusId = None
                                    break
                        else:
                            notFound = False
                            break
                        if notFound:
                            new_row = {
                                "SubmissionID": submissionID,
                                "SubmissionYear": submissionYear,
                                "SubmissionTitle": submissionTitle,
                                "SubmissionAbstract": None,
                                "firstName": firstName,
                                "middleName": middleName,
                                "lastName": lastName,
                                "isFirstAuthor": isFirstAuthor,
                                "city": None,
                                "country": country,
                                "countryCode": countryCode,
                                "institution": institution,
                                "doi": doi,
                                "authorID": authorIDReal,
                                "authorPublicationHistory": authorPublicationHistory,
                                "match": match
                            }
                            inside_data.append(new_row)
                    if break_outer_loop == False:
                        for row in inside_data:
                            openalex_data.append(row)
                    else:
                        print("Current Size:", len(openalex_data))
                        missedOut += 1
                        print("Missed Ones: ", missedOut)
                    
    openalex_df = pd.DataFrame(openalex_data)
    return openalex_df






def fetch_elsevier_data(DOI):
    API_KEY = getElsevierAPIKey()
    client = ElsClient(API_KEY)
    print(API_KEY)
    my_auth = ElsAuthor(
        uri = f"https://api.elsevier.com/content/author/orcid/{DOI}")
    # Read author data, then write to disk
    if my_auth.read(client):
        print ("my_auth.full_name: ", my_auth)
    else:
        print ("Read author failed.")
    

def extend_dataset_with_elsevier(df):
    print("Entering Here")
    for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
        doi = row["SubmissionID"]
        firstName = row["firstName"]
        lastName = row["lastName"]
        data = fetch_elsevier_data(doi)


def get_id_of_preprint_publication(data):
    data_frame = pd.DataFrame(columns=["PreprintID", "PublicationID", "Match"])
    data = pd.DataFrame(data, columns=["Preprint", "Published", "Match"])
    for index, row in tqdm.tqdm(data.iterrows(), total=data.shape[0]):
        preprintID = row["Preprint"]
        publicationID = row["Published"]
        match = row["Match"]
        preID = preprintID.get("DOI", None)
        pubID = publicationID.get("DOI", None)
        new_row = pd.DataFrame([[preID, pubID, match]], columns=["PreprintID", "PublicationID", "Match"])
        data_frame = pd.concat([data_frame, new_row], ignore_index=True)
    return data_frame
    
def main():
    
    if not os.path.exists("data_with_ids.csv"):
        data = load_data_csv("data-training.csv")
        df = get_id_of_preprint_publication(data)
        # Save the df to a csv file
        df.to_csv("data_with_ids.csv", index=False)
    else:
        df = pd.read_csv("data_with_ids.csv")
    
    if not os.path.exists("crossref_data.csv"):
        extended_dataset = extend_dataset_with_openAlex(df)
        # Save the df to a csv file
        extended_dataset.to_csv("openalex_data.csv", index=False)
    else:
        extended_dataset = pd.read_csv("openalex_data.csv")
        
    
    # Ensure the SubmissionAbstract column is of type object
    extended_dataset["SubmissionAbstract"] = extended_dataset["SubmissionAbstract"].astype(object)
    
    
    print(len(extended_dataset))
    
    if not os.path.exists("extended_dataset_v2.csv"):
        # Get all the submissionId from the extended dataset
        submissionIds = extended_dataset["SubmissionID"].tolist()
        # Remove duplicates
        print(len(submissionIds))
        submissionIds = list(set(submissionIds))
        print(len(submissionIds))
        
        print(len(extended_dataset))
        listOfIdAbstract = extend_dataset_with_crossref(submissionIds)
        for id, abstract in tqdm.tqdm(listOfIdAbstract.items()):
            print(type(abstract))
            print(type(extended_dataset.loc[extended_dataset["SubmissionID"] == id, "SubmissionAbstract"]))
            
            
            if abstract is not None:
                extended_dataset.loc[extended_dataset["SubmissionID"] == id, "SubmissionAbstract"] = abstract
            else:
                extended_dataset.drop(extended_dataset[extended_dataset["SubmissionID"] == id].index, inplace=True)
        
        extended_dataset.to_csv("extended_dataset_v2.csv", index=False)
    else:
        extended_dataset = pd.read_csv("extended_dataset_v2.csv")
    
    # Clean all the titles in the dataset
    for index, row in tqdm.tqdm(extended_dataset.iterrows(), total=extended_dataset.shape[0]):
        title = row["SubmissionTitle"]
        cleanedTitle = clean_abstract(title)
        extended_dataset.loc[index, "SubmissionTitle"] = cleanedTitle
    
    extended_dataset.to_csv("extended_dataset_v3.csv", index=False)
    
    # Printt the first row all the columns
    print(extended_dataset.iloc[0])
    


main()
