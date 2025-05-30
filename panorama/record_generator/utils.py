import re
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Set

def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if not text:
        return ""
    # Remove text within [[ ]]
    text = re.sub(r'\[\[.*?\]\]', '', text)
    text = text.replace('\n', ' ')
    # 구두점 뒤의 과도한 공백 정규화
    text = re.sub(r'([;:])\s+', r'\1 ', text)  # 구두점 뒤에 한 칸만 공백 남김
    text = re.sub(r'\s+', ' ', text)
    return text

def detect_format(root: ET.Element) -> str:
    """
    Detect the XML format by checking root tag and namespaces.
    Returns either 'pat' or 'simple' format type.
    """
    # Check if root tag directly indicates format
    root_tag = root.tag.lower()
    if root_tag.endswith('claimsdocument'):
        return 'pat'
    elif root_tag == 'us-patent-application':
        return 'simple'
        
    # If not conclusive from root tag, check namespace
    xmlns = root.get('xmlns') or ''
    if 'urn:us:gov:doc:uspto:patent' in xmlns:
        return 'pat'
        
    return 'simple'

def extract_text_from_pat_element(elem: ET.Element) -> str:
    """Extract text from patent style XML elements."""
    if elem is None:
        return ""

    text_parts = []

    if elem.text:
        text_parts.append(clean_text(elem.text))

    for child in elem:
        if any(x in child.tag.lower() for x in ['boundary-data', 'page-break', 'image']):
            if child.tail:
                text_parts.append(clean_text(child.tail))
            continue

        if any(x in child.tag.lower() for x in ['deletedtext']):
            if child.tail:
                text_parts.append(clean_text(child.tail))
        elif any(x in child.tag.lower() for x in ['insertedtext', 'ocrconfidencedata', 'claimreference', 'ins']):
            # Recursively process nested elements
            inner_text = extract_text_from_pat_element(child)
            if inner_text:
                text_parts.append(inner_text)
            if child.tail:
                text_parts.append(clean_text(child.tail))
        else:
            if child.text:
                text_parts.append(clean_text(child.text))
            if child.tail:
                text_parts.append(clean_text(child.tail))

    return ''.join(text_parts)

def extract_text_from_simple_element(elem: ET.Element) -> str:
    """Extract text from simple style XML elements."""
    if elem is None:
        return ""

    text_parts = []

    if elem.text:
        text_parts.append(clean_text(elem.text))

    for child in elem:
        if any(x in child.tag.lower() for x in ['boundary-data', 'page-break', 'confidence']):
            if child.tail:
                text_parts.append(clean_text(child.tail))
            continue

        if child.tag == 'claim-ref':
            if child.text:
                text_parts.append(clean_text(child.text))
            if child.tail:
                text_parts.append(' ' + clean_text(child.tail))
        else:
            inner_text = extract_text_from_simple_element(child)
            if inner_text:
                text_parts.append(inner_text)
            if child.tail:
                text_parts.append(clean_text(child.tail))

    return ''.join(text_parts)

def normalize_claim_spaces(claims: List[str]) -> List[str]:

    normalized_claims = []
    for claim in claims:
        # Try to split into claim number and text
        match = re.match(r'^(\d+)\.(.*?)$', claim)
        if match:
            claim_num = int(match.group(1))
            claim_text = match.group(2)
            # 첫 번째 공백만 정규화 (여러 개의 공백을 하나로)
            claim_text = re.sub(r'^\s+', ' ', claim_text)
            normalized = f"{claim_num}.{claim_text}"  # 점(.) 뒤의 공백은 유지
        else:
            # 분리할 수 없는 경우 첫 번째 공백만 정규화
            normalized = re.sub(r'^\s+', ' ', claim)
        normalized_claims.append(normalized)
    return normalized_claims

def parse_pat_claims(root: ET.Element) -> List[str]:
    """Parse patent style XML claims."""
    claims_dict: Dict[int, str] = {}
    cancelled_claims: Set[int] = set()
    processed_claims: Set[int] = set()
    
    # Find claims section
    claims_section = None
    for tag in root.iter():
        if any(x in tag.tag for x in ['ClaimList', 'ClaimSet', 'Claims']):
            claims_section = tag
            break
            
    if claims_section is None:
        return []
    
    # Process claims
    for claim in claims_section.iter():
        if not any(x in claim.tag for x in ['Claim']):
            continue

        # Get claim numbers from range if present 
        for elem in claim.iter():
            if 'ClaimNumberRange' in elem.tag:
                begin = None
                end = None
                for child in elem:
                    if 'BeginRangeNumber' in child.tag:
                        begin = int(child.text)
                    elif 'EndRangeNumber' in child.tag:
                        end = int(child.text)
                if begin and end:
                    for num in range(begin, end + 1):
                        claims_dict[num] = f"{num}. (Cancelled)"
                        cancelled_claims.add(num)
                        processed_claims.add(num)
                continue
            
        # Get claim text to check for cancellations
        claim_text = ''
        for elem in claim.iter():
            if 'ClaimText' in elem.tag:
                claim_text += extract_text_from_pat_element(elem)

        # Handle range cancellations first
        if any(x in claim_text.lower() for x in ['cancelled', 'canceled']):
            # Try all range patterns
            match = None
            patterns = [
                r'(\d+)(?:\s*\.?\s*[-–—]\s*|\s+[-–—]\s+)(\d+)\s*\.?\s*(?:\(|\[)?(?:[Cc][Aa][Nn][Cc][Ee][Ll][Ll]?[Ee]?[Dd])(?:\)|\])?'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, claim_text)
                if match:
                    break
            
            if match:
                start, end = int(match.group(1)), int(match.group(2))
                for num in range(start, end + 1):
                    claims_dict[num] = f"{num}. (Cancelled)"
                    cancelled_claims.add(num)
                    processed_claims.add(num)
                continue

        # Get claim number
        claim_num = None
        for elem in claim.iter():
            if 'ClaimNumber' in elem.tag and elem.text:
                try:
                    claim_num = int(elem.text.strip())
                    break
                except ValueError:
                    continue

        if claim_num is None or claim_num in processed_claims:
            continue

        # Check if cancelled
        status = None
        for elem in claim.iter():
            if 'ClaimStatusCategory' in elem.tag:
                status = elem.text.lower() if elem.text else None
                break

        processed_claims.add(claim_num)
        if status == 'canceled' or any(x in claim_text.lower() for x in ['cancelled', 'canceled']):
            cancelled_claims.add(claim_num)
            claims_dict[claim_num] = f"{claim_num}. (Cancelled)"
        else:
            # Get claim texts
            claim_texts = []
            for elem in claim.iter():
                if 'ClaimText' in elem.tag:
                    text = extract_text_from_pat_element(elem)
                    if text and not any(x in text.lower() for x in ['what is claimed is', 'listing of claims']):
                        claim_texts.append(clean_text(text))

            if claim_texts:
                text = ' '.join(claim_texts)
                text = re.sub(rf'^\s*{claim_num}\s*\.', '', text)
                text = re.sub(r'^\s*\([^)]+\)\s*', '', text)
                text = text.strip()
                
                if text:
                    claims_dict[claim_num] = f"{claim_num}. {text}"

    claims = [claims_dict[num] for num in sorted(claims_dict.keys())]
    return normalize_claim_spaces(claims)

def parse_simple_claims(root: ET.Element) -> List[str]:
    """Parse simple style XML claims."""
    claims_dict: Dict[int, str] = {}
    
    # Find claims section
    claims_section = None
    for tag in root.iter():
        if tag.tag == 'claims':
            claims_section = tag
            break
            
    if claims_section is None:
        return []

    # Process claims
    for claim in claims_section:
        if claim.tag != 'claim':
            continue
            
        # Get claim number
        num_attr = claim.get('num')
        if not num_attr or num_attr == "UNKNOWN":
            continue
            
        try:
            claim_num = int(num_attr)
        except ValueError:
            continue
            
        # Handle cancelled claims
        claim_text = extract_text_from_simple_element(claim)
        
        # Check for range cancellations
        if any(x in claim_text.lower() for x in ['cancelled', 'canceled']):
            match = None
            patterns = [
            r'(\d+)(?:\s*\.?\s*[-–—]\s*|\s+[-–—]\s+)(\d+)\s*\.?\s*(?:\(|\[)?(?:[Cc][Aa][Nn][Cc][Ee][Ll][Ll]?[Ee]?[Dd])(?:\)|\])?'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, claim_text)
            if match:
                start, end = int(match.group(1)), int(match.group(2))
                for num in range(start, end + 1):
                    claims_dict[num] = f"{num}. (Cancelled)"
                continue

        # Handle single cancelled claims
        if '(cancelled)' in claim_text.lower() or '(canceled)' in claim_text.lower():
            claims_dict[claim_num] = f"{claim_num}. (Cancelled)"
            continue

        # Extract claim text
        claim_texts = []
        for elem in claim:
            if elem.tag == 'claim-text':
                text = extract_text_from_simple_element(elem)
                if text and not any(x in text.lower() for x in ['what is claimed is']):
                    claim_texts.append(clean_text(text))

        if claim_texts:
            text = ' '.join(claim_texts)
            text = re.sub(rf'^\s*{claim_num}\s*\.', '', text)
            text = text.strip()
            
            if text:
                claims_dict[claim_num] = f"{claim_num}. {text.strip()}"

    claims = [claims_dict[num] for num in sorted(claims_dict.keys())]
    return normalize_claim_spaces(claims)


def parse_1st_claim(claims: List[str]) -> Optional[str]:
    """
    첫 번째 청구항의 다양한 패턴을 처리합니다:
    1. 중복된 번호 ("1. 1 1.", "6. 6.", "7. 7 7.")
    2. "claims 1)" 형식
    3. "[CLAIMS] [Claim 1]" 형식
    4. "3. 3 (Original)" 형식
    5. "1. Claims 1)" 형식
    6. "1. CLAIMS 1.A" 형식
    7. "1. CLAIMS 1." 형식
    """
    if not claims:
        return claims
        
    claim = claims[0]
    
    # 새로운 패턴: "1. CLAIMS 1." 형식
    claims_with_number_pattern = r'^\d+\.\s*CLAIMS\s*\d+\.'
    if re.match(claims_with_number_pattern, claim, re.IGNORECASE):
        content = re.sub(r'^\d+\.\s*CLAIMS\s*\d+\.', '', claim).strip()
        claims[0] = f"1. {content}"
        return claims

    # 패턴 1: 중복된 번호 패턴
    duplicate_patterns = [
        r'^\d+\.\s*\d+\s*\d+\.',
        r'^\d+\.\s*\d+\.'
    ]
    
    # 패턴 2: "claims 1)" 형식
    claims_pattern = r'^claims\s*(\d+)\)'
    
    # 패턴 3: "[CLAIMS] [Claim 1]" 형식
    bracket_pattern = r'^\[CLAIMS\]\s*\[Claim\s*(\d+)\]'
    
    # 패턴 4: "3. 3 (Original)" 형식
    original_pattern = r'^(\d+)\.\s*\d+\s*\(Original\)'
    
    # 패턴 5: "1. Claims 1)" 형식
    claims_with_num_pattern = r'^\d+\.\s*Claims\s*\d+\)'
    
    # 패턴 6: "1. CLAIMS 1.A" 형식
    claims_with_letter_pattern = r'^\d+\.\s*CLAIMS\s*\d+\.[A-Z]'

    # 중복된 번호 패턴 체크
    for pattern in duplicate_patterns:
        if re.match(pattern, claim):
            claim_num = re.match(r'(\d+)\.', claim).group(1)
            content = re.sub(pattern, '', claim).strip()
            claims[0] = f"{claim_num}. {content}"
            break
    
    # "[CLAIMS] [Claim 1]" 패턴 체크
    bracket_match = re.match(bracket_pattern, claim, re.IGNORECASE)
    if bracket_match:
        claim_num = bracket_match.group(1)
        content = re.sub(bracket_pattern, '', claim, flags=re.IGNORECASE).strip()
        claims[0] = f"{claim_num}. {content}"
    
    # "3. 3 (Original)" 패턴 체크
    original_match = re.match(original_pattern, claim)
    if original_match:
        claim_num = original_match.group(1)
        content = re.sub(r'^\d+\.\s*\d+\s*\(Original\)', '', claim).strip()
        claims[0] = f"{claim_num}. {content}"
    
    # "1. Claims 1)" 패턴 체크
    if re.match(claims_with_num_pattern, claim):
        claim_num = re.match(r'(\d+)\.', claim).group(1)
        content = re.sub(r'^\d+\.\s*Claims\s*\d+\)', '', claim).strip()
        claims[0] = f"{claim_num}. {content}"
    
    # "claims 1)" 패턴 체크
    claims_match = re.match(claims_pattern, claim, re.IGNORECASE)
    if claims_match:
        claim_num = claims_match.group(1)
        content = re.sub(claims_pattern, '', claim, flags=re.IGNORECASE).strip()
        content = re.sub(r'^[\s\.,;]+', '', content)
        claims[0] = f"{claim_num}. {content}"
    
    # 기존 로직도 유지
    splited_claim = claims[0].strip().split(".")
    if len(splited_claim) > 2 and splited_claim[2].strip().startswith("2"):
        claims[0] = "1. " + splited_claim[1] + "."
        
    return claims

def extract_claims(xml_content: str) -> List[str]:
    """Extract claims from USPTO XML content."""
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        return []

    format_type = detect_format(root)
    parsed_claims = parse_pat_claims(root) if format_type == 'pat' else parse_simple_claims(root)
    claims = parse_1st_claim(parsed_claims)
    
    # 최종적으로 반환하기 전 추가 공백 정리
    normalized_claims = []
    for c in claims:
        c = re.sub(r'\s+', ' ', c)   # 다중 공백 단일화
        c = c.strip()                # 앞뒤 공백 제거
        normalized_claims.append(c)
    
    return normalized_claims

def process_patent_xml_file(file_path: str) -> List[str]:
    """Process a patent XML file and extract claims."""
    with open(file_path, 'r', encoding='utf-8') as f:
        xml_content = f.read()
    return extract_claims(xml_content)

def validate_claims(claims: List[str]) -> bool:
    """Validate claims for common issues."""
    for enum, claim in enumerate(claims, start=1):
        if not claim.startswith(f"{enum}."):
            return False
    return True

if __name__ == "__main__":
    # Test XML file path
    xml_path = "/Users/mori/Downloads/26831_16151591_2019-06-04_CLM 2.xml"
    claims = process_patent_xml_file(xml_path)

    for c in claims:
        print(c)