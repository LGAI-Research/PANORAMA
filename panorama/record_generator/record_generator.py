import os
import json
import requests
from patent_client._sync.uspto.odp.api import ODPApi
import xml.etree.ElementTree as ET
from datetime import datetime
from utils import extract_claims, validate_claims
import time
import re

odp_api = ODPApi()
odp_api.base_url = 'https://api.uspto.gov'

USPTO_API_KEY = os.getenv("USPTO_API_KEY")

from_date = "2018-01-23T00:00:00"
to_date = "2021-01-23T00:00:00"

output_dir = "./data/record"
spec_text_dir = "./data/spec_app/text"
spec_image_dir = "./data/spec_app/image"
cited_spec_text_dir = "./data/spec_cited/text"
cited_spec_image_dir = "./data/spec_cited/image"
error_log_dir = "./data/error_report"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(spec_text_dir, exist_ok=True)
os.makedirs(spec_image_dir, exist_ok=True)
os.makedirs(cited_spec_text_dir, exist_ok=True)
os.makedirs(cited_spec_image_dir, exist_ok=True)
os.makedirs(error_log_dir, exist_ok=True)


def fetch_ctnf_documents(from_date, to_date, start_Num):
    print("Fetching CTNF documents...")
    url = "https://developer.uspto.gov/ds-api/oa_actions/v1/records"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json"
    }
    criteria = f"legacyDocumentCodeIdentifier:CTNF AND sections.grantDate:[{from_date} TO {to_date}]"
    data = {
        "criteria": criteria,
        "start": start_Num,
        "rows": 10
    }
    response = requests.post(url, headers=headers, data=data)
    data = response.json()
    
    ctnf_data = [
        {
            "title": record["inventionTitle"][0],
            "applicationNumber": record["patentApplicationNumber"][0],
            "grantDate": record["sections.grantDate"],
            "techCenter": record["techCenter"][0],
            "CTNFBodyText": record["bodyText"][0].split("Any inquiry concerning this communication")[0],
            "obsoleteDocumentIdentifier": record["obsoleteDocumentIdentifier"][0]
        }
        for record in data['response']["docs"]
    ]
    print(f"Retrieved {len(ctnf_data)} CTNF documents.")
    return ctnf_data

def fetch_grant_document(identifier):
    trimed_idendtifier = re.sub(r'[a-zA-Z]', '', identifier)

    publication_pattern = re.compile(r'^\d{11}$')
    application_pattern = re.compile(r'^\d{7,8}$')

    try:
        if publication_pattern.match(trimed_idendtifier):
            # XXXX/XXXXXXX
            print(f"Fetching grant document for publication number {trimed_idendtifier}...")
            patent = odp_api.get_documents(trimed_idendtifier)
        elif application_pattern.match(trimed_idendtifier):
            # XXXXXXX
            print(f"Fetching grant document for application number {trimed_idendtifier}...")
            patent = odp_api.get_documents(trimed_idendtifier)
        else:
            raise ValueError(f"Invalid identifier format. Please provide a valid application or publication number. input: {identifier}")
        
        application_number = patent.appl_id
        description = patent.description
        abstract = patent.abstract
        drawing = get_document_content(get_application_documents(application_number), "DRW", target_document_identifier=None, mimeTypeIdentifier="PDF")
        
        claims = [claim.text for claim in patent.claims]

        print(f"Spec, Abstract and claims retrieved for {trimed_idendtifier}.")
        return {"spec": description, "abstract": abstract, "claims": claims, "drawing": drawing}
    
    except Exception as e:
        print(f"Failed to fetch data for {trimed_idendtifier}: {e}")
        return {"spec": "", "abstract": "", "claims": [], "drawing": ""}

def extract_citations(CTNFtext):
    CTNFtext = re.sub(r'(\d),(\d)', r'\1\2', CTNFtext)
    
    unique_citations = set()
    
    slash_pattern = r'(?<!\d)(\d{4}/\d{7})(?!\d)'
    slash_matches = re.findall(slash_pattern, CTNFtext)
    
    for match in slash_matches:
        CTNFtext = CTNFtext.replace(match, '')
        unique_citations.add(match.replace('/', ''))
    
    unique_citations.update(re.findall(r'(?<!\d)\d{11}(?!\d)', CTNFtext))
    
    unique_citations.update(re.findall(r'(?<!\d)\d{7,8}(?!\d)', CTNFtext))
    
    print(f"Found {len(unique_citations)} unique citations in the CTNF text.: {unique_citations}")

    patents_cited = []
    cleaned_citations = {re.sub(r'[^\d]', '', citation) for citation in unique_citations}
    
    for id in cleaned_citations:
        cited_patent_info = fetch_grant_document(id)
        if cited_patent_info["claims"] == []:
            print(f"Patent {id} cited by examiner has no claims.")
            raise ValueError(f"Patent {id} cited by examiner has no claims")
            
        patents_cited.append({
            "referenceIdentifier": id,
            "spec": cited_patent_info["spec"],
            "abstract": cited_patent_info["abstract"],
            "claims": cited_patent_info["claims"],
            "drawing": cited_patent_info["drawing"]
        })
    
    return patents_cited

import requests
import xml.etree.ElementTree as ET

def fetch_rejected_claims(document_identifier, application_number, documents):
    print(f"{application_number}: Fetching rejected claims for document {document_identifier}...")
    target_index = next((i for i, doc in enumerate(documents) if doc["documentIdentifier"] == document_identifier), None)
    if target_index is None:
        print("Error: Document identifier not found.")
        return []

    recent_clm_doc = None
    clm_docs_on_latest_date = []
    latest_date = None
    
    for doc in documents[:target_index][::-1]:  
        if doc["documentCode"] == "CLM":
            if latest_date is None:
                latest_date = doc["officialDate"]
                clm_docs_on_latest_date.append(doc)
            elif doc["officialDate"] == latest_date:
                clm_docs_on_latest_date.append(doc)
                
    if len(clm_docs_on_latest_date) > 1:
        print(f"Error: Multiple CLM documents found on {latest_date}")
        raise ValueError(f"Multiple CLM documents found on {latest_date}")
    elif len(clm_docs_on_latest_date) == 1:
        recent_clm_doc = clm_docs_on_latest_date[0]

    if not recent_clm_doc:
        print("Error: CLM document not found before the target document.")
        raise ValueError("CLM document not found before the target document.")

    xml_download_url = next(
        (option["downloadUrl"] for option in recent_clm_doc["downloadOptionBag"] if option["mimeTypeIdentifier"] == "XML"),
        None
    )
    if not xml_download_url:
        print("Error: XML download URL not found.")
        raise ValueError("XML download URL not found.")

    headers = {"X-API-KEY": USPTO_API_KEY}
    xml_response = requests.get(xml_download_url, headers=headers)
    xml_response.raise_for_status()
    content = xml_response.content

    start_idx = content.find(b'<?xml')
    if start_idx == -1:
        print("Error: XML content not found in the file.")
        raise ValueError("XML content not found in the file.")

    xml_content = content[start_idx:].replace(b'\x00', b'').decode('utf-8', errors='ignore')

    try:
        rejected_claims = extract_claims(xml_content)
    except ET.ParseError as e:
        print("XML parsing error:", e)
        raise ValueError(f"Failed to parse XML: {e}")

    if not validate_claims(rejected_claims):
        print("Error: Invalid claim format.")
        raise ValueError("Invalid claim format.")
    
    if not rejected_claims:
        print("Error: No rejected claims found.")
        raise ValueError("No rejected claims found.")
    
    print(f"Rejected claims retrieved for application {application_number}.")
    return rejected_claims

def is_first_clm_ctnf(target_ctnf_document_id, sorted_documents):
    first_ctnf_found = None
    for doc in sorted_documents:
        if doc["documentCode"] == "CTNF":
            first_ctnf_found = doc
            break

    is_first = first_ctnf_found is not None and first_ctnf_found["documentIdentifier"] == target_ctnf_document_id
    return is_first

def get_document_content(app_documents, document_code, target_document_identifier=None, mimeTypeIdentifier="XML"):
    target_index = len(app_documents) if target_document_identifier is None else next((i for i, doc in enumerate(app_documents) if doc["documentIdentifier"] == target_document_identifier), None)
    if target_index is None:
        print("Error: Document identifier not found.")
        return []

    if document_code == "NOA":
        for doc in app_documents[:target_index]:
            if doc["documentCode"] == "NOA":
                download_url = next(
                    (option["downloadUrl"] for option in doc.get("downloadOptionBag", []) if option["mimeTypeIdentifier"] == "XML"),
                    None
                )
                if download_url:
                    headers = {"X-API-KEY": USPTO_API_KEY}
                    response = requests.get(download_url, headers=headers)
                    response.raise_for_status()
                    print(f"Retrieved NOA document.")
                    return response.content
        
        print(f"No NOA document with XML format found.")
        return ""

    recent_doc = None
    for doc in app_documents[:target_index][::-1]:  
        if doc["documentCode"].split(".")[0] == document_code:
            recent_doc = doc
            break

    download_url = next(
        (option["downloadUrl"] for option in recent_doc.get("downloadOptionBag", []) if option["mimeTypeIdentifier"] == mimeTypeIdentifier),
        None
    )

    if not download_url:
        if document_code == "DRW":
            print(f"No drawing found for {target_document_identifier}.")
        else:
            print(f"Error: {mimeTypeIdentifier} download URL not found.")
        return ""

    headers = {"X-API-KEY": USPTO_API_KEY}
    response = requests.get(download_url, headers=headers)
    response.raise_for_status()
    content = response.content
    print(f"Content retrieved.")

    return content

def parse_xml(content):
    start_idx = content.find(b'<?xml')
    if start_idx == -1:
        print(f"Error: xml content not found in the file.")
        return ""
    content = content[start_idx:].replace(b'\x00', b'').decode('utf-8', errors='ignore')

    root = ET.fromstring(content)

    def extract_all_text(element):
        text_content = []
        if element.text:
            text_content.append(element.text.strip())  
        for subelement in element:
            text_content.extend(extract_all_text(subelement))  
        if element.tail:
            text_content.append(element.tail.strip())  
        return text_content  

    document_text = ' '.join(extract_all_text(root)).strip()
    
    return document_text

def get_application_documents(app_number):
    base_url = f"https://api.uspto.gov/api/v1/patent/applications/{app_number}/documents"
    headers = {
        "X-API-KEY": USPTO_API_KEY
    }

    response = requests.get(base_url, headers=headers)
    response.raise_for_status()
    documents = response.json()['documentBag']
    
    return sorted(documents, key=lambda x: x["officialDate"])

def is_finally_rejected(sorted_documents):
    final_docs = [doc for doc in sorted_documents if doc["documentCode"] in ["NOA", "CTFR"]]
    
    if not final_docs:
        return False
        
    latest_doc = final_docs[-1]
    return latest_doc["documentCode"] == "CTFR"

def has_abst_or_spec_between_ctnf_and_noa(sorted_documents, ctnf_document_identifier, noa_document_identifier):
    ctnf_index = next((i for i, doc in enumerate(sorted_documents) 
                      if doc["documentIdentifier"] == ctnf_document_identifier), None)
    noa_index = next((i for i, doc in enumerate(sorted_documents) 
                     if doc["documentIdentifier"] == noa_document_identifier), None)
    
    if ctnf_index is None or noa_index is None:
        return False

    between_docs = sorted_documents[min(ctnf_index, noa_index):max(ctnf_index, noa_index)]

    return any(doc["documentIdentifier"] in ['ABST', 'SPEC'] for doc in between_docs)

def main(from_date, to_date):
    start_Num = 0
    
    try:
        existing_files = os.listdir(output_dir)
        
        record_numbers = []
        for file in existing_files:
            if file.startswith('rec_r'):
                try:
                    number = int(file.split('_')[1].lstrip('r0'))
                    record_numbers.append(number)
                except (ValueError, IndexError):
                    continue
        
        successful_record_count = max(record_numbers) if record_numbers else 0
        print(f"Starting with record number: {successful_record_count + 1}")
    except Exception as e:
        print(f"Error initializing record count: {e}. Starting from 0")
        successful_record_count = 0
    
    final_data = []    
    
    error_log_path = f"{error_log_dir}/error_rec_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    
    if not os.path.exists(error_log_path):
        with open(error_log_path, "w") as f:
            f.write("pre_num,rec_num,app_num,error_code,error_message,timestamp\n")
    
    while True:
        try:
            print(f"\n=== Processing batch starting from {start_Num} ===")
            ctnf_data = fetch_ctnf_documents(from_date, to_date, start_Num)
            
            if not ctnf_data:
                print(f"No more data available after start_Num {start_Num}. Ending process.")
                break

            for idx, data in enumerate(ctnf_data):
                record_number = start_Num + idx
                app_number = data["applicationNumber"]
                error_message = ""
                error_code = -1
                formatted_record_number = f"{successful_record_count + 1:05d}"

                existing_files = os.listdir(output_dir)
                if any(f"rec_r" in f and f"_{app_number}.json" in f for f in existing_files):
                    print(f"❌ {app_number} already in data")
                    error_message = f"{app_number} already in data"
                    error_code = 70
                    log_error()
                    continue

                def log_error():
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    rec_num = int(formatted_record_number) if error_code == -1 else -1
                    with open(error_log_path, "a") as f:
                        f.write(f"{record_number},{rec_num},{app_number},{error_code},\"{error_message}\",{timestamp}\n")

                sorted_documents = get_application_documents(app_number)

                start_time = time.time()

                if not is_first_clm_ctnf(data["obsoleteDocumentIdentifier"], sorted_documents):
                    error_message = f"Non-first CTNF: {data['obsoleteDocumentIdentifier']}"
                    error_code = 10
                    log_error()
                    continue

                if is_finally_rejected(sorted_documents):
                    print(f"❌ Got CTFR: {data['obsoleteDocumentIdentifier']}")
                    error_message = f"Got CTFR: {data['obsoleteDocumentIdentifier']}"
                    error_code = 20
                    log_error()
                    continue

                noa_document_identifier = next((doc["documentIdentifier"] for doc in sorted_documents if doc["documentCode"] == "NOA"), None)
                if noa_document_identifier is None:
                    print(f"❌ NOA document not found")
                    error_message = "NOA document not found"
                    error_code = 30
                    log_error()
                    continue

                try:
                    noa_body_text = parse_xml(get_document_content(sorted_documents, "NOA"))
                    if not noa_body_text:  
                        print(f"❌ NOA document exists but XML format not found")
                        error_message = "NOA document exists but XML format not found"
                        error_code = 31
                        log_error()
                        continue
                except Exception as e:
                    print(f"❌ Failed to parse NOA XML: {e}")
                    error_message = f"Failed to parse NOA XML: {e}"
                    error_code = 32
                    log_error()
                    continue

                if has_abst_or_spec_between_ctnf_and_noa(sorted_documents, data["obsoleteDocumentIdentifier"], noa_document_identifier):
                    print(f"❌ ABST or SPEC is modified between CTNF and NOA")
                    error_message = f"ABST or SPEC is modified between CTNF and NOA"
                    error_code = 40
                    log_error()
                    continue

                try:
                    patents_cited = extract_citations(data["CTNFBodyText"])
                except ValueError as e:
                    if "cited by examiner has no claims" in str(e):
                        print(f"❌ Cited patent has no claims: {e}")
                        error_message = f"Cited patent has no claims: {e}"
                        error_code = 61  
                    else:
                        print(f"❌ Failed to process cited patents: {e}")
                        error_message = f"Failed to process cited patents: {e}"
                        error_code = 60  
                    log_error()
                    continue

                metadata_url = f"https://api.uspto.gov/api/v1/patent/applications/{app_number}/meta-data"
                metadata_response = requests.get(metadata_url, headers={"X-API-KEY": USPTO_API_KEY})
                metadata_response.raise_for_status()
                metadata = {k: v for k, v in metadata_response.json()["patentFileWrapperDataBag"][0]["applicationMetaData"].items() if k != "inventorBag"}
                print(f"Metadata retrieved for application {app_number}.")
                
                try:
                    print(f"Fetching specification for application {app_number}...")
                    desc = parse_xml(get_document_content(sorted_documents, "SPEC", target_document_identifier=data["obsoleteDocumentIdentifier"]))
                except Exception as e:
                    print(f"❌ Failed to fetch specification for application {app_number}: {e}")
                    error_message = f"Failed to fetch or parse specification: {e}"
                    error_code = 51
                    log_error()
                    continue

                try:
                    print(f"Fetching abstract for application {app_number}...")
                    abstract = "ABSTRACT" + parse_xml(get_document_content(sorted_documents, "ABST", target_document_identifier=data["obsoleteDocumentIdentifier"])).split("ABSTRACT")[-1]
                except Exception as e:
                    print(f"❌ Failed to fetch abstract for application {app_number}: {e}")
                    error_message = f"Failed to fetch or parse abstract: {e}"
                    error_code = 52
                    log_error()
                    continue

                try:
                    print(f"Fetching drawing for application {app_number}...")
                    drw = get_document_content(sorted_documents, "DRW", target_document_identifier=data["obsoleteDocumentIdentifier"], mimeTypeIdentifier="PDF")
                except Exception as e:
                    print(f"❌ Failed to fetch drawing for application {app_number}: {e}")
                    error_message = f"Failed to fetch or parse drawing although it exists: {e}"
                    error_code = 53
                    log_error()
                    continue
  
                try:
                    initial_claims = fetch_rejected_claims(data["obsoleteDocumentIdentifier"], app_number, sorted_documents)
                except Exception as e:
                    print(f"❌ Failed to fetch initial claims for application {app_number}: {e}")
                    error_message = f"Failed to fetch or parse initial claims: {e}"
                    error_code = 54
                    log_error()
                    continue

                if initial_claims == []:
                    print(f"❌ Initial claims for application {app_number} is Empty")
                    error_message = "Initial claims is Empty"
                    error_code = 55
                    log_error()
                    continue          

                try:
                    final_claims = fetch_rejected_claims(noa_document_identifier, app_number, sorted_documents)
                except Exception as e:
                    print(f"❌ Failed to fetch final claims for application {app_number}: {e}")
                    error_message = f"Failed to fetch or parse final claims: {e}"
                    error_code = 56
                    log_error()
                    continue
                
                formatted_record_number = f"{successful_record_count + 1:05d}"
                spec_text_path = os.path.join("./data/spec_app/text", f"spec_txt_r{formatted_record_number}_{app_number}.txt")
                with open(spec_text_path, "w") as f:
                    f.write(desc)

                for cited_patent in patents_cited:
                    cited_spec_text_path = os.path.join("./data/spec_cited/text", f"spec_txt_{cited_patent['referenceIdentifier']}.txt")
                    with open(cited_spec_text_path, "w") as f:
                        f.write(cited_patent["spec"])
                
                if drw:
                    drw_path = os.path.join("./data/spec_app/image", f"spec_img_r{formatted_record_number}_{app_number}.pdf")
                    with open(drw_path, "wb") as f:
                        f.write(drw)
                
                for cited_patent in patents_cited:
                    cited_drw_path = os.path.join("./data/spec_cited/image", f"spec_img_{cited_patent['referenceIdentifier']}.pdf")
                    with open(cited_drw_path, "wb") as f:
                        f.write(cited_patent["drawing"])

                record = {
                    "id": successful_record_count + 1,
                    "abstract": abstract,
                    "initialClaims": initial_claims,
                    "finalClaims": final_claims,
                    "CTNFDocumentIdentifier": data["obsoleteDocumentIdentifier"],
                    "CTNFBodyText": data["CTNFBodyText"],
                    "NOABodyText": noa_body_text,
                    "applicationNumber": app_number,
                    "patentsCitedByExaminer": [
                        {"referenceIdentifier": cited_patent["referenceIdentifier"], "abstract": cited_patent["abstract"], "claims": cited_patent["claims"]} for cited_patent in patents_cited
                    ],
                    **metadata,
                }

                successful_record_count += 1
                output_path = os.path.join(output_dir, f"rec_r{formatted_record_number}_{app_number}.json")
                with open(output_path, "w") as f:
                    json.dump(record, f, indent=4)
                print(f"✅ Record {formatted_record_number} saved to {output_path}.")
                
                final_data.append(record)
                print(f"Record {formatted_record_number} processed in {time.time() - start_time:.2f} seconds.")
                
                log_error()

            start_Num += 100
            if start_Num >= 10000:
                print(f"Resetting start_Num from {start_Num} to 0")
                start_Num = 0
                
                error_log_path = f"{error_log_dir}/error_rec_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
                
                with open(error_log_path, "w") as f:
                    f.write("pre_num,rec_num,app_num,error_code,error_message,timestamp\n")
                print(f"Created new error log file: {error_log_path}")
            print(f"Moving to next batch with start_Num = {start_Num}")
            time.sleep(5)  
            
        except Exception as e:
            print(f"Error occurred in batch starting from {start_Num}: {e}")
            print("Retrying after 5 seconds...")
            time.sleep(5)
            continue
    
    return final_data

result = main(from_date, to_date)
print("All records have been processed and saved.")